#!/usr/bin/env python3
"""
Weighted Box Fusion (WBF) ensemble of multiple detection model outputs.

Merges CSV submissions from different models (e.g. YOLOv11 + RT-DETR + custom)
into a single, stronger submission using WBF.

Usage:
    python scripts/ensemble_wbf.py \
        --inputs submissions/yolo_submission.csv submissions/rtdetr_submission.csv \
        --weights 1.0 1.0 \
        --output submissions/ensemble_submission.csv \
        --iou-thr 0.55 --conf-thr 0.05
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from ensemble_boxes import weighted_boxes_fusion


# All RIVA images are 1024×1024
IMG_W = 1024
IMG_H = 1024


def load_submission(csv_path: str) -> dict[str, dict]:
    """
    Load a Kaggle-format submission CSV.
    Returns {image_filename: {"boxes": [[x1,y1,x2,y2], ...], "scores": [...], "labels": [...]}}
    Coordinates are normalised to [0, 1].
    """
    data: dict[str, dict] = defaultdict(lambda: {"boxes": [], "scores": [], "labels": []})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["image_filename"]
            cx = float(row["x"]) / IMG_W
            cy = float(row["y"]) / IMG_H
            w  = float(row["width"])  / IMG_W
            h  = float(row["height"]) / IMG_H

            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)

            data[fname]["boxes"].append([x1, y1, x2, y2])
            data[fname]["scores"].append(float(row["conf"]))
            data[fname]["labels"].append(int(row["class"]))

    return dict(data)


def main():
    parser = argparse.ArgumentParser(description="WBF ensemble of detection CSVs")
    parser.add_argument("--inputs",   nargs="+", required=True,
                        help="Paths to submission CSV files")
    parser.add_argument("--weights",  nargs="+", type=float, default=None,
                        help="Model weights for WBF (len = num inputs)")
    parser.add_argument("--output",   type=str,  default="submissions/ensemble_submission.csv")
    parser.add_argument("--iou-thr",  type=float, default=0.55,
                        help="IoU threshold for box fusion")
    parser.add_argument("--conf-thr", type=float, default=0.05,
                        help="Minimum confidence after fusion")
    parser.add_argument("--skip-box-thr", type=float, default=0.001,
                        help="Skip boxes with conf below this before WBF")
    args = parser.parse_args()

    n_models = len(args.inputs)
    if args.weights is None:
        args.weights = [1.0] * n_models
    assert len(args.weights) == n_models, "Number of weights must match number of inputs"

    # Load all submissions
    all_data = [load_submission(p) for p in args.inputs]
    print(f"Loaded {n_models} submissions:")
    for p, d in zip(args.inputs, all_data):
        total_dets = sum(len(v["boxes"]) for v in d.values())
        print(f"  {p}: {len(d)} images, {total_dets} detections")

    # Collect all unique image filenames
    all_images = set()
    for d in all_data:
        all_images.update(d.keys())
    all_images = sorted(all_images)
    print(f"\nTotal unique images: {len(all_images)}")

    # Run WBF per image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    det_id = 0
    rows = []

    for fname in all_images:
        boxes_list = []
        scores_list = []
        labels_list = []

        for d in all_data:
            if fname in d:
                boxes_list.append(d[fname]["boxes"])
                scores_list.append(d[fname]["scores"])
                labels_list.append(d[fname]["labels"])
            else:
                # Model has no detections for this image
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        # Convert to numpy
        boxes_list  = [np.array(b) if len(b) > 0 else np.zeros((0, 4)) for b in boxes_list]
        scores_list = [np.array(s) if len(s) > 0 else np.zeros((0,))   for s in scores_list]
        labels_list = [np.array(l) if len(l) > 0 else np.zeros((0,))   for l in labels_list]

        # WBF
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=args.weights,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
        )

        # Filter by confidence
        keep = fused_scores >= args.conf_thr
        fused_boxes  = fused_boxes[keep]
        fused_scores = fused_scores[keep]

        # Convert back to RIVA submission format (center x, y, width, height in pixels)
        for j in range(len(fused_boxes)):
            x1, y1, x2, y2 = fused_boxes[j]
            cx = (x1 + x2) / 2 * IMG_W
            cy = (y1 + y2) / 2 * IMG_H
            w  = (x2 - x1) * IMG_W
            h  = (y2 - y1) * IMG_H

            rows.append({
                "id":             det_id,
                "image_filename": fname,
                "class":          0,
                "x":              round(cx, 4),
                "y":              round(cy, 4),
                "width":          round(w, 4),
                "height":         round(h, 4),
                "conf":           round(float(fused_scores[j]), 6),
            })
            det_id += 1

    # Write output CSV
    fieldnames = ["id", "image_filename", "class", "x", "y", "width", "height", "conf"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Ensemble saved: {output_path}  ({det_id} detections)")


if __name__ == "__main__":
    main()
