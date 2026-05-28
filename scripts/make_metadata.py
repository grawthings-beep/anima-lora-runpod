#!/usr/bin/env python3
"""Generate an sd-scripts dataset .toml for Anima LoRA training.

sd-scripts auto-discovers images and matching sidecar .txt captions inside the
image_dir, so this only needs to emit a small TOML pointing at the train folder.
It also prints the estimated steps/epoch to help avoid over-training.
"""
import argparse
import pathlib

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

TEMPLATE = """[general]
shuffle_caption = false
caption_extension = "{caption_ext}"
keep_tokens = 0

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}
enable_bucket = true
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  num_repeats = {num_repeats}
"""


def count_images(dataset_dir):
    return [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def main():
    parser = argparse.ArgumentParser(description="Create an sd-scripts dataset.toml for Anima LoRA training.")
    parser.add_argument("--dataset-dir", default="/workspace/anima-lora/datasets/train")
    parser.add_argument("--output", default="/workspace/anima-lora/datasets/dataset.toml")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--caption-extension", default=".txt")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    images = count_images(dataset_dir)
    if not images:
        raise SystemExit(f"No images found in {dataset_dir}")

    toml = TEMPLATE.format(
        caption_ext=args.caption_extension,
        resolution=args.resolution,
        batch_size=args.batch_size,
        image_dir=dataset_dir.as_posix(),
        num_repeats=args.num_repeats,
    )

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml, encoding="utf-8")

    steps_per_epoch = (len(images) * args.num_repeats) // max(args.batch_size, 1)
    print(f"Wrote {output}")
    print(f"images={len(images)} num_repeats={args.num_repeats} resolution={args.resolution} batch_size={args.batch_size}")
    print(f"estimated steps/epoch = {steps_per_epoch} (target ~1000-2000 total; tune NUM_REPEATS/NUM_EPOCHS)")


if __name__ == "__main__":
    main()
