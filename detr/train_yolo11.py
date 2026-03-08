#!/usr/bin/env python3
"""
Train YOLOv11x on RIVA Track B for ensemble diversity with RT-DETR.

Usage:
    python scripts/train_yolo11.py --model yolo11x.pt --imgsz 1024 --batch 8 --name yolo11x_1024
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for RIVA cell detection")
    parser.add_argument("--model",    type=str,   default="yolo11x.pt",   help="yolo11n/s/m/l/x.pt")
    parser.add_argument("--data",     type=str,   default="yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs",   type=int,   default=150)
    parser.add_argument("--imgsz",    type=int,   default=1024)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--device",   type=str,   default="0")
    parser.add_argument("--name",     type=str,   default="yolo11x_1024")
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--patience", type=int,   default=30)
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] {data_yaml} not found. Run 'python scripts/convert_to_yolo.py' first.")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/detect",

        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_bias_lr=0.0,

        # Augmentation — tuned for medical microscopy
        hsv_h=0.005,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,
        erasing=0.1,

        # Close mosaic last 15 epochs
        close_mosaic=15,

        # Misc
        patience=args.patience,
        save_period=10,
        val=True,
        plots=True,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ YOLOv11 training complete!")
    print(f"   Best weights: runs/detect/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
