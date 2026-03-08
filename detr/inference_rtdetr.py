#!/usr/bin/env python3
"""
Run inference with a trained RT-DETR (or YOLOv11) model and produce
a Kaggle-format CSV submission.

Usage:
    python scripts/inference_rtdetr.py \
        --weights experiments/rtdetr_riva/weights/best.pt \
        --test-dir riva-partb-dataset/images/test \
        --output submissions/rtdetr_submission.csv \
        --imgsz 640 --conf 0.1
"""

import argparse
import csv
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="RT-DETR / YOLO inference → CSV")
    parser.add_argument("--weights",  type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="riva-partb-dataset/images/test")
    parser.add_argument("--output",   type=str, default="submissions/rtdetr_submission.csv")
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--conf",     type=float, default=0.1)
    parser.add_argument("--iou",      type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device",   type=str, default="0")
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--augment",  action="store_true", help="Test-time augmentation")
    parser.add_argument("--fix-box-size", type=int, default=100,
                        help="Force all boxes to this fixed w,h (0=disable). "
                             "RIVA GT boxes are all 100x100.")
    args = parser.parse_args()

    from ultralytics import RTDETR, YOLO

    # Auto-detect model type from weights filename
    weights_name = Path(args.weights).stem.lower()
    if "yolo" in weights_name or "yolo" in str(Path(args.weights).parent).lower():
        model = YOLO(args.weights)
    else:
        model = RTDETR(args.weights)

    test_dir = Path(args.test_dir)
    image_paths = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} test images in {test_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    det_id = 0
    rows = []

    for img_path in tqdm(image_paths, desc="Inference"):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            augment=args.augment,
            verbose=False,
        )

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # boxes.xywh → center-x, center-y, width, height  (pixel coords)
            xywh  = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for j in range(len(xywh)):
                cx = float(xywh[j, 0])
                cy = float(xywh[j, 1])

                # Force uniform box size if specified (RIVA GT is always 100x100)
                if args.fix_box_size > 0:
                    w = float(args.fix_box_size)
                    h = float(args.fix_box_size)
                else:
                    w = float(xywh[j, 2])
                    h = float(xywh[j, 3])

                rows.append({
                    "id":             det_id,
                    "image_filename": img_path.name,
                    "class":          0,
                    "x":              cx,
                    "y":              cy,
                    "width":          w,
                    "height":         h,
                    "conf":           float(confs[j]),
                })
                det_id += 1

    # Write CSV
    fieldnames = ["id", "image_filename", "class", "x", "y", "width", "height", "conf"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Submission saved: {output_path}  ({det_id} detections)")


if __name__ == "__main__":
    main()
