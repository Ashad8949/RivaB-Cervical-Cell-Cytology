#!/usr/bin/env python3
"""
Convert RIVA CSV annotations → YOLO format labels.

RIVA CSV format:  image_filename, x (center), y (center), width, height, class_name, class
YOLO label format:  class_id  cx_norm  cy_norm  w_norm  h_norm

Usage:
    python scripts/convert_to_yolo.py
"""

import os
import csv
import shutil
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path("riva-partb-dataset")
CSV_DIR      = BASE_DIR / "annotations"
IMAGE_DIR    = BASE_DIR / "images"
OUT_DIR      = Path("yolo_dataset")        # output root

# Splits to process  (csv_name, image_subfolder, split_name)
SPLITS = [
    ("train.csv", "train", "train"),
    ("val.csv",   "val",   "val"),
]

IMG_W = 1024   # all RIVA images are 1024×1024
IMG_H = 1024


def convert():
    # Create output structure
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Also create a test images symlink / copy
    test_img_src = IMAGE_DIR / "test"
    test_img_dst = OUT_DIR / "images" / "test"
    if test_img_src.exists() and not test_img_dst.exists():
        test_img_dst.symlink_to(test_img_src.resolve())

    for csv_name, img_subfolder, split in SPLITS:
        csv_path = CSV_DIR / csv_name
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping")
            continue

        # Group annotations by image
        annotations: dict[str, list] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["image_filename"]
                annotations.setdefault(fname, []).append(row)

        # Write YOLO labels + symlink images
        for fname, rows in annotations.items():
            # Image symlink
            src_img = (IMAGE_DIR / img_subfolder / fname).resolve()
            dst_img = OUT_DIR / "images" / split / fname
            if src_img.exists() and not dst_img.exists():
                dst_img.symlink_to(src_img)

            # Label file  (same stem, .txt extension)
            label_path = OUT_DIR / "labels" / split / Path(fname).with_suffix(".txt")
            with open(label_path, "w") as lf:
                for row in rows:
                    cx = float(row["x"]) / IMG_W
                    cy = float(row["y"]) / IMG_H
                    w  = float(row["width"]) / IMG_W
                    h  = float(row["height"]) / IMG_H

                    # Clip to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w  = max(0.001, min(1.0, w))
                    h  = max(0.001, min(1.0, h))

                    # Single class → class_id = 0
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"[{split}] {len(annotations)} images, "
              f"{sum(len(v) for v in annotations.values())} annotations → "
              f"{OUT_DIR / 'labels' / split}")

    # Write dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: cell\n"
    )
    print(f"\nDataset YAML written to {yaml_path}")


if __name__ == "__main__":
    convert()
