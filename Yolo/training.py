"""
RIVA Part B - Training Module for YOLOv11x-Swin (Transformer Backbone)
Based on successful YOLOv8x configuration, optimized for Swin backbone

Can be run directly: python training.py

Assumes fold directories already exist with structure:
  fold_0/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
"""

import os
import argparse
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import TrainingConfig, ModelConfig, DataConfig, set_seed


class ImprovedModelTrainer:
    """Train YOLO models with Swin Transformer backbone settings."""
    
    def __init__(
        self, 
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
    def train_fold(self, fold_idx: int, fold_dir: str) -> Optional[Dict]:
        """Train a single fold and return metrics."""
        print(f"\n{'='*80}")
        print(f"TRAINING FOLD {fold_idx} - YOLOv11x-Swin (Transformer Backbone)")
        print(f"{'='*80}")
        
        # Initialize model
        model = YOLO(self.model_config.primary_model)
        print(f"  Loaded {self.model_config.primary_model}")
        print(f"  Backbone: Swin Transformer")
        print(f"  Training with image size: {self.training_config.imgsz}px")
        print(f"  Batch size: {self.training_config.batch} (reduced for Swin)")
        print(f"  Mosaic augmentation: {self.training_config.mosaic}")
        print(f"  Learning rate: {self.training_config.lr0} (lower for transformer)")
        print(f"  Weight decay: {self.training_config.weight_decay} (higher for transformer)")
        print(f"  Warmup epochs: {self.training_config.warmup_epochs} (longer for transformer)")
        
        # Train model
        results = model.train(
            data=os.path.join(fold_dir, "data.yaml"),
            
            # === Image and batch ===
            epochs=self.training_config.epochs,
            imgsz=self.training_config.imgsz,
            batch=self.training_config.batch,
            workers=self.training_config.workers,
            
            # === Optimizer (tuned for Swin Transformer) ===
            optimizer=self.training_config.optimizer,
            lr0=self.training_config.lr0,
            lrf=self.training_config.lrf,
            weight_decay=self.training_config.weight_decay,
            momentum=self.training_config.momentum,
            
            # === Scheduler ===
            cos_lr=self.training_config.cos_lr,
            warmup_epochs=self.training_config.warmup_epochs,
            warmup_momentum=self.training_config.warmup_momentum,
            warmup_bias_lr=self.training_config.warmup_bias_lr,
            
            # === Augmentation (CRITICAL: Mosaic ENABLED) ===
            augment=True,
            hsv_h=self.training_config.hsv_h,
            hsv_s=self.training_config.hsv_s,
            hsv_v=self.training_config.hsv_v,
            degrees=self.training_config.degrees,
            translate=self.training_config.translate,
            scale=self.training_config.scale,
            shear=self.training_config.shear,
            perspective=self.training_config.perspective,
            flipud=self.training_config.flipud,
            fliplr=self.training_config.fliplr,
            bgr=self.training_config.bgr,
            
            # CRITICAL: Mosaic enabled
            mosaic=self.training_config.mosaic,
            mixup=self.training_config.mixup,
            copy_paste=self.training_config.copy_paste,
            close_mosaic=self.training_config.close_mosaic,
            auto_augment=self.training_config.auto_augment,
            erasing=self.training_config.erasing,
            crop_fraction=self.training_config.crop_fraction,
            
            # === Regularization ===
            label_smoothing=self.training_config.label_smoothing,
            dropout=self.training_config.dropout,
            nbs=self.training_config.nbs,
            
            # === Loss weights ===
            box=self.training_config.box,
            cls=self.training_config.cls,
            dfl=self.training_config.dfl,
            
            # === Detection settings ===
            single_cls=self.training_config.single_cls,
            overlap_mask=self.training_config.overlap_mask,
            mask_ratio=self.training_config.mask_ratio,
            
            # === Validation ===
            patience=self.training_config.patience,
            save=self.training_config.save,
            save_period=self.training_config.save_period,
            val=self.training_config.val,
            plots=self.training_config.plots,
            cache=self.training_config.cache,
            
            # === Misc ===
            device=self.training_config.device,
            seed=self.training_config.seed,
            deterministic=self.training_config.deterministic,
            verbose=self.training_config.verbose,
            
            # === Output ===
            project=os.path.join(self.data_config.work_dir, "runs"),
            name=f"fold_{fold_idx}",
            exist_ok=True,
        )
        
        # Validate and collect metrics
        best_model_path = os.path.join(
            self.data_config.work_dir, 
            "runs", 
            f"fold_{fold_idx}", 
            "weights", 
            "best.pt"
        )
        
        if os.path.exists(best_model_path):
            return self._validate_model(fold_idx, fold_dir, best_model_path)
        else:
            print(f"  Warning: Best model not found at {best_model_path}")
            return None
    
    def _validate_model(
        self, 
        fold_idx: int, 
        fold_dir: str, 
        model_path: str
    ) -> Dict:
        """Validate trained model and return metrics."""
        print(f"\n{'='*60}")
        print(f"VALIDATING FOLD {fold_idx}")
        print(f"{'='*60}")
        
        val_model = YOLO(model_path)
        
        metrics = val_model.val(
            data=os.path.join(fold_dir, "data.yaml"),
            imgsz=self.training_config.imgsz,
            conf=0.001,
            iou=0.5,
            device=self.training_config.device,
            split="val"
        )
        
        # Extract metrics safely
        precision = float(
            metrics.box.p.mean() 
            if isinstance(metrics.box.p, np.ndarray) 
            else metrics.box.p
        )
        recall = float(
            metrics.box.r.mean() 
            if isinstance(metrics.box.r, np.ndarray) 
            else metrics.box.r
        )
        
        fold_metrics = {
            'fold': fold_idx,
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),
            'precision': precision,
            'recall': recall,
            'model_path': model_path
        }
        
        self._print_fold_results(fold_metrics)
        return fold_metrics
    
    def _print_fold_results(self, metrics: Dict):
        """Print fold validation results."""
        print(f"\n  FOLD {metrics['fold']} VALIDATION RESULTS:")
        print(f"  mAP50:     {metrics['mAP50']:.4f}")
        print(f"  mAP50-95:  {metrics['mAP50_95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")


class ImprovedTrainingPipeline:
    """Orchestrate training across all folds."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig
    ):
        self.trainer = ImprovedModelTrainer(model_config, training_config, data_config)
        self.data_config = data_config
        
    def train_all_folds(self, fold_dirs: List[str]) -> List[Dict]:
        """Train all folds and return metrics."""
        all_metrics = []
        
        for fold_idx, fold_dir in enumerate(fold_dirs):
            try:
                metrics = self.trainer.train_fold(fold_idx, fold_dir)
                if metrics:
                    all_metrics.append(metrics)
                else:
                    print(f"\n  Warning: Fold {fold_idx} training did not produce metrics")
            except Exception as e:
                print(f"\n  Error training fold {fold_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save and print summary
        if all_metrics:
            self._save_metrics(all_metrics)
            self._print_summary(all_metrics)
        else:
            print("\n  No successful fold training!")
        
        return all_metrics
    
    def _save_metrics(self, metrics: List[Dict]):
        """Save training metrics to CSV."""
        metrics_df = pd.DataFrame(metrics)
        output_path = os.path.join(
            self.data_config.work_dir, 
            "training_metrics_swin.csv"
        )
        metrics_df.to_csv(output_path, index=False)
        print(f"\n  Metrics saved to {output_path}")
    
    def _print_summary(self, metrics: List[Dict]):
        """Print training summary."""
        metrics_df = pd.DataFrame(metrics)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE - K-FOLD SUMMARY (YOLOv11x-Swin)")
        print("="*80)
        print(metrics_df.to_string(index=False))
        print(f"\n{'='*80}")
        print(f"Average mAP50-95: {metrics_df['mAP50_95'].mean():.4f} +/- {metrics_df['mAP50_95'].std():.4f}")
        print(f"Average mAP50:    {metrics_df['mAP50'].mean():.4f} +/- {metrics_df['mAP50'].std():.4f}")
        print(f"Average Precision: {metrics_df['precision'].mean():.4f} +/- {metrics_df['precision'].std():.4f}")
        print(f"Average Recall:    {metrics_df['recall'].mean():.4f} +/- {metrics_df['recall'].std():.4f}")
        print("="*80)
        
        # Best fold
        best_fold = metrics_df.loc[metrics_df['mAP50_95'].idxmax()]
        print(f"\nBEST FOLD: Fold {int(best_fold['fold'])}")
        print(f"   mAP50-95: {best_fold['mAP50_95']:.4f}")
        print(f"   Model: {best_fold['model_path']}")
        print("="*80)


def find_fold_directories(work_dir: str) -> List[str]:
    """Auto-detect fold directories."""
    fold_dirs = []
    
    if not os.path.exists(work_dir):
        return fold_dirs
    
    for item in sorted(os.listdir(work_dir)):
        if item.startswith("fold_") and os.path.isdir(os.path.join(work_dir, item)):
            fold_path = os.path.join(work_dir, item)
            # Verify it has the required structure
            if os.path.exists(os.path.join(fold_path, "data.yaml")):
                fold_dirs.append(fold_path)
    
    return fold_dirs


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='RIVA Training - YOLOv11x-Swin (Transformer Backbone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all folds with default settings
  python training.py
  
  # Train specific fold
  python training.py --fold 0
  
  # Custom settings
  python training.py --work_dir my_exp --epochs 200 --batch 2
  
  # Use different GPU
  python training.py --device 1
        """
    )
    parser.add_argument('--work_dir', type=str, default='yolov11x-swin',
                       help='Working directory containing fold_* directories')
    parser.add_argument('--model', type=str, default='yolo11x-swin.pt',
                       help='YOLO model to use (yolo11x-swin.pt, yolo11x.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size (default: 1280)')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size (default: 2, reduced for Swin Transformer)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--fold', type=int, default=None,
                       help='Train specific fold only (0-4), or all if not specified')
    parser.add_argument('--lr0', type=float, default=0.0002,
                       help='Initial learning rate (default: 0.0002, lower for Swin Transformer)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create configs
    model_config = ModelConfig(primary_model=args.model)
    training_config = TrainingConfig(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0
    )
    data_config = DataConfig(work_dir=args.work_dir)
    
    # Print header
    print("="*80)
    print("RIVA YOLOv11x-Swin TRAINING - TRANSFORMER BACKBONE")
    print("="*80)
    print(f"\nModel:       {args.model}")
    print(f"Backbone:    Swin Transformer")
    print(f"Work Dir:    {args.work_dir}")
    print(f"Image Size:  {args.imgsz}px")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch:       {args.batch} (reduced for Swin)")
    print(f"LR:          {args.lr0} (lower for transformer)")
    print(f"Device:      cuda:{args.device}")
    
    print(f"\nKEY SWIN TRANSFORMER SETTINGS:")
    print(f"  Mosaic augmentation: ENABLED (0.3)")
    print(f"  Rotation: Conservative (+/-10 deg)")
    print(f"  Image size: 1280px")
    print(f"  Learning rate: {args.lr0} (lower for transformer)")
    print(f"  Weight decay: 0.05 (higher for transformer)")
    print(f"  Warmup: 10 epochs (longer for transformer)")
    print("="*80)
    
    # Find fold directories
    fold_dirs = find_fold_directories(args.work_dir)
    
    if not fold_dirs:
        print(f"\n  No fold directories found in {args.work_dir}")
        print(f"\nExpected structure:")
        print(f"  {args.work_dir}/")
        print(f"    fold_0/")
        print(f"      data.yaml")
        print(f"      images/train/")
        print(f"      images/val/")
        print(f"      labels/train/")
        print(f"      labels/val/")
        print(f"    fold_1/")
        print(f"    ...")
        return
    
    print(f"\n  Found {len(fold_dirs)} fold(s):")
    for fold_dir in fold_dirs:
        print(f"    - {fold_dir}")
    
    # Train specific fold or all folds
    if args.fold is not None:
        if args.fold >= len(fold_dirs):
            print(f"\n  Fold {args.fold} not found (available: 0-{len(fold_dirs)-1})")
            return
        fold_dirs = [fold_dirs[args.fold]]
        print(f"\n  Training only fold {args.fold}")
    else:
        print(f"\n  Training all {len(fold_dirs)} folds")
    
    print("\n" + "="*80)
    
    # Run training
    pipeline = ImprovedTrainingPipeline(model_config, training_config, data_config)
    all_metrics = pipeline.train_all_folds(fold_dirs)
    
    if not all_metrics:
        print("\n  No models were successfully trained!")
        print("\nTroubleshooting:")
        print("  1. Check that fold directories exist and have correct structure")
        print("  2. Verify data.yaml files are properly formatted")
        print("  3. Check GPU availability (nvidia-smi)")
        print("  4. Review error messages above")
        return
    
    print("\n  Training complete! Models saved to:")
    for metric in all_metrics:
        print(f"    - {metric['model_path']}")
    
    print(f"\n  Next steps:")
    print(f"  1. Review training plots in {args.work_dir}/runs/")
    print(f"  2. Run inference with best models")
    print(f"  3. Generate ensemble predictions")


if __name__ == "__main__":
    main()
