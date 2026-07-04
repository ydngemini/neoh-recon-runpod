# Neoh GPU Reconstruction — RunPod Serverless

A **RunPod Serverless** GPU worker that turns a property's captured photos into a
walkable 3D **Gaussian splat** (`.splat`). It runs the same pipeline as the AWS
Batch worker (`../reconstruction/`) — **COLMAP poses → nerfstudio splatfacto →
PlayCanvas splat-transform** — but is driven by RunPod's serverless contract
instead of AWS Batch. Scales to zero; you pay per-second only while a house
reconstructs.

**Why this exists:** the AWS Batch path needs EC2 **G/VT Spot** vCPU quota
(L-3819A6DF), which has been stuck at 0 with an open support case for days. RunPod
gives us GPUs immediately, with no quota gate. This worker is a drop-in compute
backend — it writes the finished `.splat` to the **same S3 bucket** the app
already reads, so nothing downstream changes.

All licensing is commercial-OK: COLMAP (BSD), nerfstudio/gsplat/tinycudann
(Apache-2.0), splat-transform (MIT). Do **not** swap in DUSt3R / INRIA-3DGS
(non-commercial).

```
CaptureWizard → POST /api/crm/reconstruction-jobs
  → reconstruction_worker → RunPodProvider:
        upload photos → s3://<bucket>/recon-inputs/<job>/
        POST https://api.runpod.ai/v2/<endpoint>/run          ──► handler.py on the GPU:
            {"input": {input_s3, output_s3, iters}}                 pull images (S3 or URLs)
        poll GET .../status/<id> until COMPLETED                    pipeline.sh:
                                                                      COLMAP poses (ns-process-data)
                                                                      train splatfacto (Apache 3DGS)
                                                                      export .ply → splat-transform → .splat
                                                                    upload .splat → recon-outputs/<job>/
        download model.splat from S3
  → _store_splat (S3): splats/<media_id>.splat → property_media kind='splat'
  → resolver tier 3 → "Step inside" walk
```

## Files
- `handler.py` — RunPod serverless handler. Owns transport (S3 / presigned URL /
  base64) and the job contract; shells out to `pipeline.sh` for compute.
- `pipeline.sh` — the transport-free compute core (images dir → `.splat`).
- `Dockerfile` — CUDA 12.1-devel + COLMAP + nerfstudio splatfacto + splat-transform
  + the RunPod SDK. Arch list covers Ampere-24G (8.6) **and** Ada-24G (8.9).
- `requirements.txt` — `runpod`, `boto3`, `requests`.
- `.runpod/hub.json` + `.runpod/tests.json` — RunPod **Hub** listing + CI test.
- `test_input.json` — payload for a local `python handler.py` run.

## Job contract

**Input** (`job["input"]`):

| field | type | notes |
|---|---|---|
| `input_s3` | string | `s3://bucket/recon-inputs/<job>` — pull every object under the prefix. *(image source)* |
| `image_urls` | string[] | download each URL (presigned OK). *(image source — use instead of `input_s3`)* |
| `output_s3` | string | `s3://bucket/recon-outputs/<job>/model.splat` — upload result. *(sink)* |
| `output_put_url` | string | presigned **PUT** URL to receive the `.splat`. *(sink)* |
| `return_splat_b64` | bool | base64 the `.splat` into the reply. *(sink)* |
| `iters` | int | splatfacto iterations (default `RECON_ITERS` env or 7000). |
| `selftest` | bool | skip GPU work; emit a synthetic demo room `.splat` (boot check). |

Provide **one image source** and **at least one sink**.

**Output:** `{"gaussians": int, "bytes": int, "splat_s3"?, "splat_put"?, "splat_b64"?, "disclosure"}`.
Failures return `{"error": "..."}` (RunPod marks the job `FAILED`).

## Deploy A — RunPod Hub (from GitHub, the "Create Listing" flow)

The Hub builds and hosts your endpoint from a GitHub repo. It indexes **releases,
not commits**, so you must cut a release.

1. Push this directory to its own GitHub repo (see **Standalone repo** below).
2. In the RunPod console → **Hub → Create Listing**, connect that GitHub repo.
   The Hub reads `.runpod/hub.json` (GPU pool, disk, env inputs) and
   `.runpod/tests.json` (the `boot-selftest` CI test).
3. Create a **GitHub release** (e.g. `v1`). The Hub picks it up, builds the image,
   runs the selftest, and marks the listing `Pending → Published`.
4. Deploy an endpoint from the listing. Note the **Endpoint ID**.

## Deploy B — Manual (build + push + create endpoint)

Build needs an NVIDIA toolchain host (tinycudann compiles with `nvcc`):
```bash
docker build -t <registry>/neoh-recon-runpod:v1 infra/reconstruction-runpod
docker push  <registry>/neoh-recon-runpod:v1
```
Then RunPod console → **Serverless → New Endpoint**: set the image, a 24 GB GPU
(RTX 4090 / L4 / A10), **Container Disk ≥ 40 GB**, and — critically — raise the
**Execution Timeout** (default is 10 min; a house takes 20–60 min → set ~3600 s).
Add env `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` as
**secrets** if you use S3 mode.

## Standalone repo (for the Hub)

The Hub needs its own repo. Extract this subtree without disturbing the monorepo:
```bash
git subtree split -P infra/reconstruction-runpod -b runpod-worker
git push git@github.com:<you>/neoh-recon-runpod.git runpod-worker:main
```

## Local smoke test (no GPU)
```bash
pip install runpod boto3 requests
python handler.py            # auto-loads test_input.json → selftest demo splat
```
`selftest` skips COLMAP/training, so it validates the handler + transport wiring
without a GPU. A real reconstruction needs the CUDA image and 8+ photos.

## Wire it into Oracle

Add this provider to `backend/reconstruction_providers.py` and register it in
`_PROVIDERS` as `"runpod"`. It mirrors `AwsBatchProvider` (same bucket, same
`recon-inputs/<job>` and `recon-outputs/<job>/model.splat` keys), so `_store_splat`
and everything downstream is unchanged — and it **cancels** the RunPod job on
timeout (the AWS path leaks orphaned jobs on timeout).

```python
class RunPodProvider(ReconstructionProvider):
    """RunPod Serverless GPU reconstruction — the no-AWS-GPU-quota path.
    Uploads the capture to the existing recon S3 bucket, submits to /run, polls
    /status, then downloads the .splat the worker wrote back to S3.

    Env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, RECON_S3_BUCKET, AWS_REGION.
    """
    name = "runpod"

    def available(self) -> tuple[bool, str]:
        miss = [v for v in ("RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID", "RECON_S3_BUCKET")
                if not os.environ.get(v)]
        if miss:
            return (False, "set " + ", ".join(miss) + " (deploy infra/reconstruction-runpod)")
        return (True, "")

    async def reconstruct(self, images: list[Path], work_dir: Path) -> Path:
        if not images:
            raise ProviderError("no capture images provided")
        return await asyncio.to_thread(self._run_blocking, images, work_dir)

    def _run_blocking(self, images: list[Path], work_dir: Path) -> Path:
        import time, uuid as _uuid, boto3, requests
        api = os.environ["RUNPOD_API_KEY"]
        endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
        bucket = os.environ["RECON_S3_BUCKET"]
        region = os.environ.get("AWS_REGION", "us-east-1")
        timeout = int(os.environ.get("RECON_RUNPOD_TIMEOUT", "3600"))
        base = f"https://api.runpod.ai/v2/{endpoint}"
        hdr = {"Authorization": f"Bearer {api}", "Content-Type": "application/json"}

        s3 = boto3.client("s3", region_name=region)
        job_key = _uuid.uuid4().hex
        in_prefix = f"recon-inputs/{job_key}"
        out_key = f"recon-outputs/{job_key}/model.splat"
        for p in images:
            s3.upload_file(str(p), bucket, f"{in_prefix}/{p.name}")

        body = {"input": {"input_s3": f"s3://{bucket}/{in_prefix}",
                          "output_s3": f"s3://{bucket}/{out_key}"},
                "policy": {"executionTimeout": timeout * 1000,
                           "ttl": (timeout + 600) * 1000}}
        r = requests.post(f"{base}/run", json=body, headers=hdr, timeout=60)
        r.raise_for_status()
        job_id = r.json()["id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(15)
            st = requests.get(f"{base}/status/{job_id}", headers=hdr, timeout=30).json()
            status = st.get("status")
            if status == "COMPLETED":
                if (st.get("output") or {}).get("error"):
                    raise ProviderError(f"RunPod job error: {st['output']['error']}")
                break
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise ProviderError(f"RunPod job {status}: {st.get('output') or st.get('error')}")
        else:
            requests.post(f"{base}/cancel/{job_id}", headers=hdr, timeout=30)
            raise ProviderError("RunPod reconstruction did not finish within budget")

        out = work_dir / "model.splat"
        s3.download_file(bucket, out_key, str(out))
        return out
```

Then set the backend env and roll the service:
```
RECONSTRUCTION_PROVIDER=runpod
RUNPOD_API_KEY=...              # RunPod → Settings → API Keys
RUNPOD_ENDPOINT_ID=...          # from the deployed endpoint
RECON_S3_BUCKET=<existing recon bucket>   # reuse infra/terraform/reconstruction.tf's bucket
AWS_REGION=us-east-1
```
The worker's own S3 access uses the `AWS_*` secrets you set **on the RunPod
endpoint** (it runs outside AWS). Presigned-URL mode avoids giving RunPod any AWS
creds — pass `image_urls` + `output_put_url` instead.

## Notes / hardening
- **Execution timeout**: RunPod's default is 10 min. The provider sends
  `policy.executionTimeout` per request; also raise the endpoint default so the
  console doesn't kill long jobs.
- **Result retention**: async `/run` results are kept only ~30 min after
  completion — fine here because the splat is persisted to S3, not carried in the
  reply.
- **GPU pool**: 24 GB (Ada `8.9` or Ampere `8.6`) is plenty for a house at 7000
  iters. The image is built for both arches.
- **Selftest** is the cheap boot check; it writes a synthetic room splat with no
  GPU/photos and is what the Hub CI test runs.
