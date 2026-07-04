# Neoh reconstruction — RunPod Serverless GPU worker.
# COLMAP + nerfstudio splatfacto + PlayCanvas splat-transform, driven by handler.py.
#
# Base is CUDA 12.1 **runtime** (not devel): the build compiles only two tiny C
# wheels (pyliblzfse, fpsample) with gcc — nothing needs nvcc (tinycudann is not
# installed; gsplat/nerfacc ship prebuilt CUDA wheels that use this base's runtime
# libs). Dropping the devel toolkit takes the pushed image from ~7.2 GB to ~2.8 GB
# so RunPod serverless workers finish the cold-start pull before they're preempted
# (the 7.2 GB devel image never left "initializing" on contended GPUs).
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
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
RUN npm install -g @playcanvas/splat-transform && npm cache clean --force \
    && pip3 install --upgrade pip

# Torch (CUDA 12.1) then nerfstudio. gsplat/nerfacc install as prebuilt wheels and
# use the CUDA runtime libs in this base — no nvcc/devel toolkit required.
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
WORKDIR /
COPY handler.py /handler.py

# RunPod serverless workers start by running the handler; it blocks on the queue.
CMD ["python3", "-u", "/handler.py"]
