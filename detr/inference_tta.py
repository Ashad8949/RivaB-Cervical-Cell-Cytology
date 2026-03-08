#!/usr/bin/env python3
"""
Multi-scale Test-Time Augmentation (TTA) inference with WBF fusion.

Instead of relying on ultralytics' basic --augment flag, this runs inference
at multiple scales and with flips, then fuses all detections per image using WBF.

This typically adds +2-4% mAP over single-scale inference.

Usage:
    # Single model, multi-scale TTA
    python scripts/inference_tta.py \
        --weights runs/detect/rtdetr_l_1024/weights/best.pt \
        --output submissions/rtdetr_l_1024_tta.csv

    # Custom scales
    python scripts/inference_tta.py \
        --weights runs/detect/rtdetr_l_1024/weights/best.pt \
        --scales 640 960 1024 1280 \
        --output submissions/rtdetr_l_1024_tta.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion


IMG_W = 1024
IMG_H = 1024


def run_inference(model, img_path, imgsz, conf, iou, device, flip=None):
    """Run inference, optionally with flips. Returns list of (x1,y1,x2,y2,conf) in pixel coords."""
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return []

    if flip == "lr":
        img = cv2.flip(img, 1)  # horizontal flip
    elif flip == "ud":
        img = cv2.flip(img, 0)  # vertical flip

    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]

            # Un-flip coordinates
            if flip == "lr":
                x1_new = IMG_W - x2
                x2_new = IMG_W - x1
                x1, x2 = x1_new, x2_new
            elif flip == "ud":
                y1_new = IMG_H - y2
                y2_new = IMG_H - y1
                y1, y2 = y1_new, y2_new

            detections.append([x1, y1, x2, y2, float(confs[j])])

    return detections


def main():
    parser = argparse.ArgumentParser(description="Multi-scale TTA inference with WBF")
    parser.add_argument("--weights",    type=str, required=True)
    parser.add_argument("--test-dir",   type=str, default="riva-partb-dataset/images/test")
    parser.add_argument("--output",     type=str, default="submissions/rtdetr_tta.csv")
    parser.add_argument("--scales",     type=int, nargs="+", default=[640, 960, 1024, 1280],
                        help="Inference scales for TTA")
    parser.add_argument("--conf",       type=float, default=0.05, help="Low conf for TTA (WBF filters later)")
    parser.add_argument("--iou",        type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--wbf-iou",    type=float, default=0.55, help="WBF IoU threshold")
    parser.add_argument("--wbf-conf",   type=float, default=0.10, help="Min confidence after WBF")
    parser.add_argument("--device",     type=str, default="0")
    parser.add_argument("--use-flips",  action="store_true", default=True,
                        help="Include horizontal and vertical flip TTA")
    parser.add_argument("--no-flips",   action="store_true", help="Disable flip TTA")
    parser.add_argument("--fix-box-size", type=int, default=100,
                        help="Force all boxes to this fixed w,h (0=disable). "
                             "RIVA GT boxes are all 100x100, so fixing predictions to 100x100 "
                             "removes size noise and improves IoU at strict thresholds.")
    args = parser.parse_args()

    if args.no_flips:
        args.use_flips = False

    from ultralytics import RTDETR, YOLO

    # Auto-detect model type
    weights_str = str(args.weights).lower()
    if "yolo" in weights_str:
        model = YOLO(args.weights)
    else:
        model = RTDETR(args.weights)

    test_dir = Path(args.test_dir)
    image_paths = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} test images in {test_dir}")

    # Build TTA configs: (scale, flip)
    tta_configs = []
    for scale in args.scales:
        tta_configs.append((scale, None))
        if args.use_flips:
            tta_configs.append((scale, "lr"))
            tta_configs.append((scale, "ud"))

    print(f"TTA configs: {len(tta_configs)} passes per image "
          f"(scales={args.scales}, flips={'on' if args.use_flips else 'off'})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    det_id = 0
    rows = []

    for img_path in tqdm(image_paths, desc="TTA Inference"):
        # Collect detections from all TTA passes
        all_boxes = []  # list of arrays, one per TTA config
        all_scores = []
        all_labels = []

        for scale, flip in tta_configs:
            dets = run_inference(model, img_path, scale, args.conf, args.iou, args.device, flip)
            if len(dets) == 0:
                all_boxes.append(np.zeros((0, 4)))
                all_scores.append(np.array([]))
                all_labels.append(np.array([]))
                continue

            dets = np.array(dets)
            # Normalize to [0,1] for WBF
            boxes_norm = dets[:, :4].copy()
            boxes_norm[:, [0, 2]] /= IMG_W
            boxes_norm[:, [1, 3]] /= IMG_H
            boxes_norm = np.clip(boxes_norm, 0, 1)

            all_boxes.append(boxes_norm)
            all_scores.append(dets[:, 4])
            all_labels.append(np.zeros(len(dets), dtype=int))

        # WBF fusion
        if all(len(s) == 0 for s in all_scores):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=args.wbf_iou,
            skip_box_thr=args.wbf_conf,
            conf_type="avg",
        )

        # Convert back to pixel center coords
        for j in range(len(fused_boxes)):
            x1, y1, x2, y2 = fused_boxes[j]
            cx = (x1 + x2) / 2 * IMG_W
            cy = (y1 + y2) / 2 * IMG_H

            # Force uniform box size if specified (RIVA GT is always 100x100)
            if args.fix_box_size > 0:
                w = float(args.fix_box_size)
                h = float(args.fix_box_size)
            else:
                w  = (x2 - x1) * IMG_W
                h  = (y2 - y1) * IMG_H

            rows.append({
                "id":             det_id,
                "image_filename": img_path.name,
                "class":          0,
                "x":              float(cx),
                "y":              float(cy),
                "width":          float(w),
                "height":         float(h),
                "conf":           float(fused_scores[j]),
            })
            det_id += 1

    # Write CSV
    fieldnames = ["id", "image_filename", "class", "x", "y", "width", "height", "conf"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Submission saved: {output_path}  ({det_id} detections)")
    print(f"   Scales: {args.scales}, Flips: {args.use_flips}")
    print(f"   WBF IoU: {args.wbf_iou}, Min conf: {args.wbf_conf}")


if __name__ == "__main__":
    main()
