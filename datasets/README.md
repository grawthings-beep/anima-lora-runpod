# Dataset

Put training images here when the Pod is running:

```text
/workspace/anima-lora/datasets/train
```

Use one sidecar caption file per image:

```text
001.png
001.txt
002.png
002.txt
```

Then run:

```bash
/opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir /workspace/anima-lora/datasets/train
```

It writes `metadata.csv` in DiffSynth format:

```csv
image,prompt
001.png,"your caption"
```
