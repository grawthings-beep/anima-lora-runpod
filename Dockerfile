# syntax=docker/dockerfile:1.7

# RunPod's PyTorch image keeps CUDA/Python aligned with the platform and avoids
# rebuilding torch inside GitHub Actions.
ARG BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DIFFSYNTH_HOME=/opt/DiffSynth-Studio

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
    && git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git "${DIFFSYNTH_HOME}" \
    && python -m pip install -e "${DIFFSYNTH_HOME}" \
    && python -m pip install \
      "huggingface_hub[hf_transfer]" \
      accelerate \
      modelscope \
      pandas \
      peft \
      pillow \
      safetensors \
      sentencepiece \
      toml

COPY config/ /opt/runpod-anima-lora/config/
COPY scripts/ /opt/runpod-anima-lora/scripts/
RUN chmod +x /opt/runpod-anima-lora/scripts/*.sh

WORKDIR /workspace/anima-lora
ENTRYPOINT ["/opt/runpod-anima-lora/scripts/start.sh"]
