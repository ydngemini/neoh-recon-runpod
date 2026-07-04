# Neoh reconstruction — RunPod Serverless GPU worker.
# Same pipeline as the AWS Batch image (COLMAP + nerfstudio splatfacto + PlayCanvas
# splat-transform) but driven by the RunPod serverless handler instead of run.sh.
#
# tiny-cuda-nn (pulled by nerfstudio) is compiled from source and needs nvcc, so
# this must be a CUDA -devel base. TORCH_CUDA_ARCH_LIST covers Ampere 24G (8.6:
# A10/A5000) AND Ada 24G (8.9: RTX 4090/L4/L40) so the same image runs on either
# RunPod pool. Build on a machine with the NVIDIA toolchain.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
      python3 python3-pip python3-dev \
      colmap ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
# NOTE: Ubuntu's apt `nodejs` is Node 12 — too old for splat-transform (optional
# chaining). NodeSource gives Node 20.

# .ply -> .splat converter (MIT).
RUN npm install -g @playcanvas/splat-transform && pip3 install --upgrade pip

# Torch (CUDA 12.1) then nerfstudio (pulls tinycudann/gsplat — Apache-2.0).
RUN pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install nerfstudio

# RunPod SDK + transport deps (boto3 for S3, requests for URLs).
COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

RUN mkdir -p /work
COPY pipeline.sh /usr/local/bin/pipeline.sh
RUN chmod +x /usr/local/bin/pipeline.sh

# Handler at container root + CMD running it from root — the convention the RunPod
# Hub validator resolves through the Dockerfile to confirm runpod.serverless.start().
COPY handler.py /handler.py

# RunPod serverless workers start by running the handler; it blocks on the queue.
CMD ["python3", "-u", "/handler.py"]
