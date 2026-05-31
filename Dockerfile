# syntax=docker/dockerfile:1.7

# RunPod's PyTorch image keeps CUDA/Python aligned with the platform and avoids
# rebuilding torch inside GitHub Actions. PyTorch 2.8 + cu12.8 supports sm_120
# (RTX 50-series / Blackwell).
ARG BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Pin sd-scripts for reproducibility. Override at build time with --build-arg if needed.
# 068bcd7 = kohya-ss/sd-scripts main @ 2026-05-18 (includes Anima LoRA fixes).
ARG SD_SCRIPTS_REF=068bcd7ffe76b2cd5012fb680a2c94e295398bbc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SD_SCRIPTS_HOME=/opt/sd-scripts

RUN apt-get update && apt-get install -y --no-install-recommends \
      aria2 \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      jq \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && git clone https://github.com/kohya-ss/sd-scripts.git "${SD_SCRIPTS_HOME}" \
    && cd "${SD_SCRIPTS_HOME}" \
    && git checkout "${SD_SCRIPTS_REF}" \
    # torch/torchvision/xformers come from the base image (cu128, sm_120-capable);
    # drop any pins in requirements.txt that would downgrade/override them.
    && sed -i -E '/^(torch|torchvision|xformers)([=<>!~ ]|$)/d' requirements.txt \
    && python -m pip install -r requirements.txt \
    && python -m pip install "huggingface_hub[hf_transfer]" hf_transfer jupyterlab

COPY config/ /opt/runpod-anima-lora/config/
COPY scripts/ /opt/runpod-anima-lora/scripts/
RUN chmod +x /opt/runpod-anima-lora/scripts/*.sh

WORKDIR /workspace/anima-lora
ENTRYPOINT ["/opt/runpod-anima-lora/scripts/start.sh"]
