# RunPod Steps

## 1. Push This Repo

Create a GitHub repo, then from this folder run:

```powershell
git init
git add .
git commit -m "Add Anima LoRA RunPod trainer"
git branch -M main
git remote add origin https://github.com/grawthings-beep/anima-lora-runpod.git
git push -u origin main
```

GitHub Actions will build the GHCR image.

## 2. Create RunPod Secrets

Create:

```text
HF_TOKEN
CIVITAI_TOKEN
```

The Civitai token is required for WAI-ANIMA. The HF token is recommended for Anima support files and tokenizers.

## 3. Create Template

Template values:

```text
Name: anima-lora-runpod
Type: Pod
Compute type: Nvidia GPU
Container image: ghcr.io/grawthings-beep/anima-lora-runpod:cuda12.8
Container disk: 40 GB
Volume disk / Network Volume: 80 GB+
Volume mount path: /workspace
```

Environment variables:

```text
WORKSPACE_DIR=/workspace/anima-lora
MODEL_ROOT=/workspace/anima-lora
DOWNLOAD_MODELS=1
RUN_DEP_CHECK=1
AUTO_TRAIN=0
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
```

## 4. Launch Pod

Open the Pod terminal. First boot downloads:

```text
wai_anima_2859702.safetensors
qwen_3_06b_base.safetensors
qwen_image_vae.safetensors
```

They stay on `/workspace` while the Pod/volume exists.

## 5. Upload Dataset

Put files in:

```text
/workspace/anima-lora/datasets/train
```

Each image should have a caption file:

```text
001.png
001.txt
```

## 6. Train

```bash
/opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir /workspace/anima-lora/datasets/train
/opt/runpod-anima-lora/scripts/train_lora.sh
```

Output:

```text
/workspace/anima-lora/outputs/anima_lora
```
