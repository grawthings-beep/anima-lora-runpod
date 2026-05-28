# RunPod Steps (sd-scripts Anima LoRA)

Quick end-to-end checklist.

## 1. Build the image

1. Push to `main` (or run the `Build GHCR image` workflow manually).
2. Wait for GitHub Actions to finish.
3. In GitHub Packages, set the `anima-lora-runpod` package to **Public** (one-time), or configure a RunPod registry secret.

## 2. Create the Pod

- Container image: `ghcr.io/grawthings-beep/anima-lora-runpod:cuda12.8`
- Volume mount path: `/workspace`
- GPU: any cu12.8-capable card (RTX 50-series / Blackwell supported).
- Env vars (see README). Put `HF_TOKEN` and `CIVITAI_TOKEN` in RunPod Secrets.

On first boot the container downloads the 3 model files and runs an environment check, then idles.

## 3. Upload the dataset

Copy images + matching `.txt` captions to:

```
/workspace/anima-lora/datasets/train
```

## 4. Train

```
/opt/runpod-anima-lora/scripts/train_lora.sh
```

or override hyperparameters inline, e.g.:

```
LORA_RANK=16 NUM_REPEATS=8 NUM_EPOCHS=4 LEARNING_RATE=1e-4 \
/opt/runpod-anima-lora/scripts/train_lora.sh
```

Watch the printed `steps/epoch`; target ~1,000-2,000 total steps.

## 5. Use the LoRA

Output: `/workspace/anima-lora/outputs/anima_lora.safetensors`

Copy it to your ComfyUI `models/loras/anima` folder and load with an Anima workflow. DiT-only LoRAs need no conversion.

## Troubleshooting

- **LoRA has no effect in ComfyUI** -> check the ComfyUI console for `lora key not loaded`. DiT-only sd-scripts output should not trigger this. If you trained text-encoder LoRA too, convert with `networks/convert_anima_lora_to_comfy.py`.
- **Training too slow** -> confirm `cache_latents` / `cache_text_encoder_outputs` are active and reduce NUM_REPEATS x NUM_EPOCHS.
- **bitsandbytes / sm_120 error** -> set `OPTIMIZER_TYPE=AdamW`.
- **NaN loss** -> ensure PyTorch >= 2.5 (the base image is 2.8).
