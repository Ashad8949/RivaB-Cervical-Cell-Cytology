"""
RIVA Part B - IMPROVED Configuration Module
Based on successful YOLOv8x settings with optimizations for YOLOv11x-Swin
(Swin Transformer backbone)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import random
import numpy as np


@dataclass
class ModelConfig:
    """Model selection configuration."""
    models_to_try: List[str] = field(default_factory=lambda: ['yolo11x.yaml', 'yolo11x.pt', 'yolo8x.pt'])
    primary_model: str = 'yolo11x.yaml'
    swin_variant: str = 'swin_base_patch4_window12_384_in22k'
    swin_pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters - OPTIMIZED FOR YOLOv11x-Swin (Transformer Backbone)."""
    
    # === CRITICAL CHANGES FOR SWIN TRANSFORMER BACKBONE ===
    # Image size: Match YOLOv8x's successful 1280px
    imgsz: int = 1280
    
    # Epochs: Moderate length
    epochs: int = 150
    
    # Batch: Lower for Swin Transformer (heavier backbone, more VRAM)
    batch: int = 2
    workers: int = 8
    patience: int = 50
    
    # === OPTIMIZER - Tuned for Swin Transformer backbone ===
    optimizer: str = 'AdamW'
    lr0: float = 0.0002  # Lower LR for transformer backbone (was 0.0005)
    lrf: float = 0.01    # Keep this - works well
    weight_decay: float = 0.05  # Higher weight decay for transformers (was 0.0005)
    momentum: float = 0.937
    
    # === SCHEDULER - Longer warmup for transformer backbone ===
    cos_lr: bool = True
    warmup_epochs: int = 10  # Transformers benefit from longer warmup (was 5)
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.01  # Gentler warmup for Swin (was 0.1)
    
    # === AUGMENTATION - KEY INSIGHT: ENABLE MOSAIC ===
    # YOLOv8x uses mosaic=0.3 successfully - medical images CAN handle it
    hsv_h: float = 0.01   # Keep conservative
    hsv_s: float = 0.4    # YOLOv8x value (increase from 0.3)
    hsv_v: float = 0.3    # Keep YOLOv8x value
    
    # Geometric augmentations - more conservative like YOLOv8x
    degrees: float = 10.0  # Much less rotation (was 90!)
    translate: float = 0.1  # YOLOv8x value
    scale: float = 0.3     # YOLOv8x value (increase from 0.2)
    shear: float = 2.0     # YOLOv8x value
    perspective: float = 0.0  # Keep disabled
    flipud: float = 0.0    # YOLOv8x: no vertical flip
    fliplr: float = 0.5    # YOLOv8x: horizontal flip OK
    bgr: float = 0.0
    
    # === CRITICAL: ENABLE MOSAIC (YOLOv8x uses 0.3) ===
    mosaic: float = 0.3    # Re-enable! (was 0.0)
    mixup: float = 0.0     # Keep disabled
    copy_paste: float = 0.0  # Keep disabled
    close_mosaic: int = 20  # YOLOv8x value - disable mosaic in last 20 epochs
    auto_augment: Optional[str] = None
    erasing: float = 0.0
    crop_fraction: float = 1.0
    
    # === REGULARIZATION - Keep minimal ===
    label_smoothing: float = 0.0
    dropout: float = 0.0
    nbs: int = 64
    
    # === LOSS WEIGHTS - Standard YOLO ===
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
    # === DETECTION SETTINGS ===
    single_cls: bool = True
    overlap_mask: bool = True
    mask_ratio: int = 4
    
    # === VALIDATION ===
    val: bool = True
    save_period: int = 10
    plots: bool = True
    save: bool = True
    cache: bool = False
    
    # === REPRODUCIBILITY ===
    deterministic: bool = True
    seed: int = 42
    device: int = 0
    verbose: bool = True


@dataclass
class InferenceConfig:
    """Inference and ensemble configuration."""
    # Multi-scale inference - adjust for Swin backbone with 1280 base size
    scales: List[int] = field(default_factory=lambda: [1024, 1280, 1536])
    
    # Detection thresholds - match YOLOv8x's ultra-low threshold
    conf_threshold: float = 0.0001  # YOLOv8x value (was 0.05)
    iou_threshold: float = 0.3
    max_detections: int = 2000  # YOLOv8x value (was 1000)
    
    # Test-time augmentation
    augment: bool = True
    
    # Weighted Boxes Fusion parameters
    wbf_iou_threshold: float = 0.4
    wbf_skip_threshold: float = 0.0001  # Match YOLOv8x ultra-low threshold
    wbf_conf_type: str = 'avg'
    
    # Post-processing thresholds
    final_thresholds: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15]
    )


@dataclass
class DataConfig:
    """Dataset paths and fold configuration."""
    root_dir: str = "riva-partb-dataset"
    work_dir: str = "yolov11x-swin"
    n_folds: int = 5
    seed: int = 42
    
    @property
    def img_dir(self) -> str:
        return os.path.join(self.root_dir, "images")
    
    @property
    def ann_dir(self) -> str:
        return os.path.join(self.root_dir, "annotations")
    
    @property
    def train_csv(self) -> str:
        return os.path.join(self.ann_dir, "train.csv")
    
    @property
    def val_csv(self) -> str:
        return os.path.join(self.ann_dir, "val.csv")


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("RIVA IMPROVED PIPELINE - YOLOv11x-Swin (Transformer Backbone)")
        print("=" * 80)
        print(f"\nModel: {self.model.primary_model}")
        print(f"Backbone: Swin Transformer")
        print(f"Image Size: {self.training.imgsz}px")
        print(f"Epochs: {self.training.epochs}")
        print(f"Batch Size: {self.training.batch} (reduced for Swin)")
        print(f"Initial LR: {self.training.lr0} (lower for transformer)")
        print(f"Weight Decay: {self.training.weight_decay} (higher for transformer)")
        print(f"Warmup Epochs: {self.training.warmup_epochs} (longer for transformer)")
        print(f"K-Folds: {self.data.n_folds}")
        
        print(f"\nKEY CHANGES FOR SWIN TRANSFORMER:")
        print(f"  - Model: yolo11x.pt -> yolo11x-swin.pt (Swin backbone)")
        print(f"  - Batch size: 4 -> 2 (Swin uses more VRAM)")
        print(f"  - Learning rate: 0.0005 -> 0.0002 (transformers need lower LR)")
        print(f"  - Weight decay: 0.0005 -> 0.05 (standard for transformers)")
        print(f"  - Warmup: 5 -> 10 epochs (transformers need longer warmup)")
        print(f"  - Warmup bias LR: 0.1 -> 0.01 (gentler warmup)")
        print(f"  - Mosaic: ENABLED at 0.3")
        print(f"  - Rotation: +/-10 deg (conservative)")
        
        print(f"\nAugmentation Settings:")
        print(f"  - Mosaic: {self.training.mosaic} (ENABLED - close at epoch {self.training.close_mosaic})")
        print(f"  - Rotation: +/-{self.training.degrees} deg (conservative)")
        print(f"  - Scale: {self.training.scale}")
        print(f"  - HSV: h={self.training.hsv_h}, s={self.training.hsv_s}, v={self.training.hsv_v}")
        print(f"  - Flips: LR={self.training.fliplr}, UD={self.training.flipud}")
        
        print(f"\nInference Settings:")
        print(f"  - Multi-scale: {self.inference.scales}")
        print(f"  - Conf threshold: {self.inference.conf_threshold}")
        print(f"  - Max detections: {self.inference.max_detections}")
        print(f"  - WBF skip threshold: {self.inference.wbf_skip_threshold}")
        print(f"  - TTA: {'ENABLED' if self.inference.augment else 'DISABLED'}")
        
        print("\nRATIONALE:")
        print("  Swin Transformer backbone provides stronger feature extraction")
        print("  via shifted window self-attention. Key considerations:")
        print("  1) Lower LR & higher weight decay (standard transformer training)")
        print("  2) Longer warmup to stabilize attention layers")
        print("  3) Smaller batch size due to higher memory footprint")
        print("  4) Mosaic augmentation still beneficial")
        
        print("\nExpected Improvements over CNN backbone:")
        print("  - Better feature representations (self-attention)")
        print("  - Improved detection of small/overlapping cells")
        print("  - Stronger generalization from transformer architecture")
        print("  - Higher recall (ultra-low confidence thresholds)")
        print("=" * 80)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def get_swin_config() -> PipelineConfig:
    """
    Get configuration optimized for YOLOv11x-Swin Transformer backbone.
    Uses lower LR, higher weight decay, longer warmup.
    """
    config = PipelineConfig()
    # Already set to Swin-optimized values by default
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
