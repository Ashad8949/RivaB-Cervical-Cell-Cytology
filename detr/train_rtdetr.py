#!/usr/bin/env python3
"""
Train RT-DETR on RIVA Track B cell detection.

RT-DETR (Real-Time DEtection TRansformer) from ultralytics provides a production-
quality DETR implementation with deformable attention, proper two-stage refinement,
and IoU-aware classification — everything the custom head lacks.

Usage:
    # First convert data:
    python scripts/convert_to_yolo.py

    # Then train:
    python scripts/train_rtdetr.py                         # RT-DETR-L (default)
    python scripts/train_rtdetr.py --model rtdetr-x.pt     # RT-DETR-X (larger)
    python scripts/train_rtdetr.py --resume                # resume last run
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR for RIVA cell detection")
    parser.add_argument("--model",   type=str,   default="rtdetr-l.pt",  help="Model variant (rtdetr-l.pt or rtdetr-x.pt)")
    parser.add_argument("--data",    type=str,   default="yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs",  type=int,   default=80)
    parser.add_argument("--imgsz",   type=int,   default=640,            help="Image size (640 or 1024)")
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--device",  type=str,   default="0")
    parser.add_argument("--name",    type=str,   default="rtdetr_riva")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--patience",type=int,   default=20,             help="Early stopping patience")
    args = parser.parse_args()

    # Ensure YOLO-format data exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] {data_yaml} not found. Run 'python scripts/convert_to_yolo.py' first.")
        return

    from ultralytics import RTDETR

    model = RTDETR(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="experiments",

        # Optimiser
        optimizer="AdamW",
        lr0=0.0001,          # initial LR
        lrf=0.01,            # final LR factor (cosine to lr0 * lrf)
        weight_decay=0.0001,
        warmup_epochs=5,
        warmup_bias_lr=0.0,

        # Augmentation (moderate for medical images)
        hsv_h=0.01,          # hue shift
        hsv_s=0.3,           # saturation shift
        hsv_v=0.2,           # value shift
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.3,
        mosaic=0.5,          # mosaic helps with dense small objects
        mixup=0.1,
        copy_paste=0.1,

        # Misc
        patience=args.patience,
        save_period=10,       # checkpoint every 10 epochs
        val=True,
        plots=True,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print(f"   Best weights: experiments/{args.name}/weights/best.pt")
    print(f"   Last weights: experiments/{args.name}/weights/last.pt")
    return results


if __name__ == "__main__":
    main()
