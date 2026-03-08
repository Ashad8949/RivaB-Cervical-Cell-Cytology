#!/usr/bin/env python3
"""
Improved RT-DETR training for RIVA Track B — targets 0.55+ mAP.

Key improvements over v1:
  - imgsz=1024 (native resolution, cells are 100px in 1024px images)
  - More epochs (150) with close_mosaic=20 for clean convergence
  - Tuned augmentation for medical microscopy (less hue, more geometric)
  - Support for RT-DETR-L and RT-DETR-X
  - Multi-scale training via rect=False
  - Better LR schedule

Usage:
    # RT-DETR-L at 1024 (recommended first)
    python scripts/train_rtdetr_v2.py --model rtdetr-l.pt --imgsz 1024 --batch 4 --name rtdetr_l_1024

    # RT-DETR-X at 1024 (larger model)
    python scripts/train_rtdetr_v2.py --model rtdetr-x.pt --imgsz 1024 --batch 2 --name rtdetr_x_1024

    # Resume training
    python scripts/train_rtdetr_v2.py --resume --name rtdetr_l_1024
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR v2 for RIVA cell detection")
    parser.add_argument("--model",    type=str,   default="rtdetr-l.pt",  help="rtdetr-l.pt or rtdetr-x.pt")
    parser.add_argument("--data",     type=str,   default="yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs",   type=int,   default=150)
    parser.add_argument("--imgsz",    type=int,   default=1024,           help="Image size (1024 = native)")
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--device",   type=str,   default="0")
    parser.add_argument("--name",     type=str,   default="rtdetr_l_1024")
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--patience", type=int,   default=30,             help="Early stopping patience")
    args = parser.parse_args()

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
        project="runs/detect",

        # Optimiser — lower LR for larger images
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,            # cosine decay to lr0 * 0.01
        weight_decay=0.0001,
        warmup_epochs=5,
        warmup_bias_lr=0.0,

        # Augmentation — tuned for medical microscopy
        hsv_h=0.005,          # minimal hue (staining is meaningful)
        hsv_s=0.4,            # moderate saturation (staining variation)
        hsv_v=0.3,            # moderate brightness
        degrees=15.0,         # rotation (cells can be any orientation)
        translate=0.15,       # translation
        scale=0.4,            # scale variation (important for detection)
        shear=2.0,            # slight shear
        fliplr=0.5,
        flipud=0.5,           # cells are rotation-invariant
        mosaic=0.8,           # mosaic helps with dense small objects
        mixup=0.15,           # mild mixup
        copy_paste=0.15,      # copy-paste aug for object detection
        erasing=0.1,          # random erasing

        # Close mosaic for last N epochs (cleaner convergence)
        close_mosaic=20,

        # Misc
        patience=args.patience,
        save_period=10,
        val=True,
        plots=True,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print(f"   Best weights: runs/detect/{args.name}/weights/best.pt")
    print(f"   Last weights: runs/detect/{args.name}/weights/last.pt")
    return results


if __name__ == "__main__":
    main()
