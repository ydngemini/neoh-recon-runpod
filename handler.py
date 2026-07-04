#!/usr/bin/env python3
"""RunPod serverless handler — Gaussian-splat property reconstruction.

Runs the SAME recipe as the Neoh AWS Batch worker (infra/reconstruction/run.sh):
COLMAP poses (nerfstudio ns-process-data) -> train splatfacto (Apache-2.0 3DGS)
-> export .ply -> PlayCanvas splat-transform -> antimatter15 .splat. Only the
transport/orchestration changes: RunPod serverless /run + /status instead of
AWS Batch submit_job / describe_jobs. The compute core lives in pipeline.sh.

Input (job["input"]):
  # image source — provide exactly one
  input_s3        "s3://bucket/recon-inputs/<job>"   pull every object under the prefix
  image_urls      ["https://.../1.jpg", ...]         download each (presigned URLs OK)
  # result sink — provide at least one
  output_s3       "s3://bucket/recon-outputs/<job>/model.splat"
  output_put_url  "https://...signed-PUT..."         HTTP PUT the .splat there
  return_splat_b64  true                              base64 the .splat into the reply
  # tuning
  iters           int    splatfacto iterations (default $RECON_ITERS or 7000)
  # ops
  selftest        true   skip GPU work; emit a synthetic demo splat (boot check)

Output: {"gaussians": int, "bytes": int, "splat_s3"?: str, "splat_put"?: bool,
         "splat_b64"?: str, "selftest"?: bool, "disclosure": str}
On failure returns {"error": "..."} so RunPod marks the job FAILED gracefully.

S3 modes require AWS creds in the worker env (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) — set them as RunPod endpoint secrets.
Presigned-URL modes need no AWS creds at all.
"""
from __future__ import annotations

import base64
import collections
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

DISCLOSURE = (
    "AI-generated 3D reconstruction from photos; geometry may be incomplete or "
    "inaccurate. Not a measured survey or a substitute for an in-person showing."
)
MIN_IMAGES = 8
PIPELINE = "/usr/local/bin/pipeline.sh"


# --- transport: image source -----------------------------------------------
def _split_s3(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"not an s3:// uri: {uri}")
    return p.netloc, p.path.lstrip("/")


def _pull_s3_prefix(uri: str, dest: Path) -> int:
    import boto3  # lazy: only S3 mode needs the SDK

    bucket, prefix = _split_s3(uri)
    s3 = boto3.client("s3")
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            name = os.path.basename(key) or f"img_{n:04d}"
            s3.download_file(bucket, key, str(dest / name))
            n += 1
    return n


def _pull_urls(urls: list[str], dest: Path) -> int:
    if not isinstance(urls, list) or not urls:
        raise ValueError("image_urls must be a non-empty list")
    for i, url in enumerate(urls):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        with open(dest / f"img_{i:04d}{ext}", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return len(urls)


def _gather(job_input: dict, images: Path) -> int:
    if job_input.get("input_s3"):
        return _pull_s3_prefix(job_input["input_s3"], images)
    if job_input.get("image_urls"):
        return _pull_urls(job_input["image_urls"], images)
    raise ValueError("provide an image source: input_s3 or image_urls")


# --- transport: result sink -------------------------------------------------
def _push_s3(path: Path, uri: str) -> None:
    import boto3

    bucket, key = _split_s3(uri)
    boto3.client("s3").upload_file(
        str(path), bucket, key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )


def _push_put_url(path: Path, url: str) -> None:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=300)
    r.raise_for_status()


def _emit(job_input: dict, splat: Path) -> dict:
    out: dict = {}
    sank = False
    if job_input.get("output_s3"):
        _push_s3(splat, job_input["output_s3"])
        out["splat_s3"] = job_input["output_s3"]
        sank = True
    if job_input.get("output_put_url"):
        _push_put_url(splat, job_input["output_put_url"])
        out["splat_put"] = True
        sank = True
    if job_input.get("return_splat_b64"):
        out["splat_b64"] = base64.b64encode(splat.read_bytes()).decode("ascii")
        sank = True
    if not sank:
        raise ValueError("provide a result sink: output_s3, output_put_url, or return_splat_b64")
    return out


# --- synthetic splat for selftest (no GPU, no images) -----------------------
def _row(px, py, pz, sx, sy, sz, r, g, b, a=255) -> bytes:
    # 32-byte gsplat row: pos 3xf32 | scale 3xf32 | rgba 4xu8 | rot 4xu8 (identity).
    return struct.pack("<3f3f", px, py, pz, sx, sy, sz) + bytes((r, g, b, a, 255, 128, 128, 128))


def _write_demo_splat(path: Path, w=4.0, h=2.6, d=4.0, step=0.12) -> Path:
    rows = bytearray()
    y = 0.0
    while y <= h + 1e-6:  # four walls
        x = -w / 2
        while x <= w / 2 + 1e-6:
            rows.extend(_row(x, y, -d / 2, 0.05, 0.05, 0.012, 196, 184, 168))
            rows.extend(_row(x, y, d / 2, 0.05, 0.05, 0.012, 196, 184, 168))
            x += step
        y += step
    a = -w / 2
    while a <= w / 2 + 1e-6:  # floor
        b = -d / 2
        while b <= d / 2 + 1e-6:
            rows.extend(_row(a, 0.0, b, 0.05, 0.012, 0.05, 150, 140, 128))
            b += step
        a += step
    path.write_bytes(rows)
    return path


# --- pipeline ---------------------------------------------------------------
def _run_pipeline(images: Path, splat: Path, iters: str) -> tuple[int, str]:
    """Run pipeline.sh, tee its output to the RunPod worker log, keep an error tail."""
    env = {**os.environ, "RECON_ITERS": iters}
    proc = subprocess.Popen(
        [PIPELINE, str(images), str(splat)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail: collections.deque[str] = collections.deque(maxlen=80)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)  # -> visible live in RunPod worker logs
        tail.append(line)
    proc.wait()
    return proc.returncode, "".join(tail)


def handler(job):
    job_input = job.get("input") or {}
    work = Path(tempfile.mkdtemp(prefix="recon-"))
    try:
        splat = work / "model.splat"

        if job_input.get("selftest"):
            runpod.serverless.progress_update(job, "selftest: writing demo splat")
            _write_demo_splat(splat)
            result = _emit(job_input, splat)
            data = splat.read_bytes()
            return {"gaussians": len(data) // 32, "bytes": len(data),
                    "selftest": True, "disclosure": DISCLOSURE, **result}

        images = work / "images"
        images.mkdir()
        runpod.serverless.progress_update(job, "downloading capture images")
        n = _gather(job_input, images)
        if n < MIN_IMAGES:
            return {"error": f"need >={MIN_IMAGES} images for reconstruction, got {n}"}

        iters = str(int(job_input.get("iters") or os.environ.get("RECON_ITERS", "7000")))
        runpod.serverless.progress_update(job, f"reconstructing: COLMAP + splatfacto ({iters} iters, {n} images)")
        rc, tail = _run_pipeline(images, splat, iters)
        if rc != 0:
            return {"error": f"reconstruction failed (exit {rc})", "log": tail[-1500:]}
        if not splat.is_file():
            return {"error": "pipeline produced no .splat", "log": tail[-1500:]}

        runpod.serverless.progress_update(job, "uploading splat")
        result = _emit(job_input, splat)
        data = splat.read_bytes()
        return {"gaussians": len(data) // 32, "bytes": len(data),
                "iters": int(iters), "images": n,
                "disclosure": DISCLOSURE, **result}
    except Exception as e:  # graceful FAILED with a readable reason
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
