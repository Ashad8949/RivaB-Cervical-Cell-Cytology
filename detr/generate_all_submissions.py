#!/usr/bin/env python3
"""
Generate all available TTA submissions and WBF ensemble.

Usage:
    python scripts/generate_all_submissions.py
"""

import subprocess
from pathlib import Path


def run(cmd, description=""):
    print(f"\n{'='*60}\n  {description}\n  $ {cmd}\n{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    base = Path("/teamspace/studios/this_studio")

    models = []
    for name, path, scales in [
        ("rtdetr_640",    "runs/detect/experiments/rtdetr_riva/weights/best.pt", [640, 960, 1024]),
        ("rtdetr_l_1024", "runs/detect/rtdetr_l_1024/weights/best.pt",          [960, 1024, 1280]),
        ("rtdetr_x_1024", "runs/detect/rtdetr_x_1024/weights/best.pt",          [960, 1024, 1280]),
        ("yolo11x_1024",  "runs/detect/yolo11x_1024/weights/best.pt",           [960, 1024, 1280]),
    ]:
        w = base / path
        if w.exists():
            models.append((name, str(w), scales))

    print(f"Found {len(models)} models: {[m[0] for m in models]}")

    subs = []
    for name, weights, scales in models:
        out = f"submissions/{name}_tta.csv"
        scales_str = " ".join(str(s) for s in scales)
        ok = run(
            f"python scripts/inference_tta.py --weights {weights} "
            f"--scales {scales_str} --output {out} "
            f"--conf 0.05 --wbf-iou 0.55 --wbf-conf 0.08 --device 0",
            f"TTA: {name}"
        )
        if ok and Path(out).exists():
            subs.append(out)

    if len(subs) > 1:
        inputs = " ".join(subs)
        weights = " ".join(["1.0"] * len(subs))
        run(
            f"python scripts/ensemble_wbf.py --inputs {inputs} "
            f"--weights {weights} --output submissions/final_ensemble.csv "
            f"--iou-thr 0.55 --conf-thr 0.05",
            "WBF Ensemble"
        )

    print("\nSubmissions:")
    for f in sorted(Path("submissions").glob("*.csv")):
        lines = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:40s} {lines:6d} detections")


if __name__ == "__main__":
    main()
