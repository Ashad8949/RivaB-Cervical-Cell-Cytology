"""
RIVA Part B - Standalone Data Preparation Module
Can be run directly: python data_preparation.py

Usage:
  python data_preparation.py --root_dir riva-dataset-partb --work_dir yolov11x
  python data_preparation.py --root_dir /path/to/data --work_dir output --n_folds 5
"""

import os
import shutil
import yaml
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import DataConfig, set_seed


class DatasetLoader:
    """Load and combine training and validation datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.combined_df: pd.DataFrame = None
        self.image_to_annotations: Dict[str, pd.DataFrame] = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load and combine train and validation CSVs."""
        print("\nLoading dataset...")
        
        train_csv = self.config.train_csv
        val_csv = self.config.val_csv
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV not found: {train_csv}")
        if not os.path.exists(val_csv):
            raise FileNotFoundError(f"Val CSV not found: {val_csv}")
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        self.combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Create mapping from image to annotations
        image_files = self.combined_df['image_filename'].unique()
        for img_name in image_files:
            self.image_to_annotations[img_name] = self.combined_df[
                self.combined_df['image_filename'] == img_name
            ]
        
        self._print_statistics()
        return self.combined_df, self.image_to_annotations
    
    def _print_statistics(self):
        """Print dataset statistics."""
        n_images = self.combined_df['image_filename'].nunique()
        n_annotations = len(self.combined_df)
        avg_cells = n_annotations / n_images
        
        print(f"Dataset Statistics:")
        print(f"  Images: {n_images:,}")
        print(f"  Annotations: {n_annotations:,}")
        print(f"  Avg cells/image: {avg_cells:.1f}")


class FoldCreator:
    """Create stratified K-fold splits."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def create_folds(self, combined_df: pd.DataFrame) -> List[Dict]:
        """Create stratified K-fold splits based on cell counts."""
        print(f"\nCreating {self.config.n_folds}-fold splits...")
        
        image_files = combined_df['image_filename'].unique()
        
        # Stratify by cell count for balanced folds
        image_cell_counts = combined_df.groupby('image_filename').size()
        bins = pd.qcut(
            image_cell_counts.values, 
            q=5, 
            labels=False, 
            duplicates='drop'
        )
        
        # K-Fold with stratification
        skf = StratifiedKFold(
            n_splits=self.config.n_folds, 
            shuffle=True, 
            random_state=self.config.seed
        )
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_files, bins)):
            folds.append({
                'fold': fold_idx,
                'train_images': image_files[train_idx],
                'val_images': image_files[val_idx]
            })
            
            print(f"  Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
        
        return folds


class FoldPreparer:
    """Prepare fold directories with images and labels."""
    
    def __init__(self, config: DataConfig, image_to_annotations: Dict[str, pd.DataFrame]):
        self.config = config
        self.image_to_annotations = image_to_annotations
        
    def prepare_fold(
        self, 
        fold_idx: int, 
        train_images: np.ndarray, 
        val_images: np.ndarray
    ) -> str:
        """Prepare a single fold with images and YOLO labels."""
        fold_dir = os.path.join(self.config.work_dir, f"fold_{fold_idx}")
        
        # Create directory structure
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(fold_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "labels", split), exist_ok=True)
        
        # Process training images
        self._process_images(
            fold_dir, 
            train_images, 
            "train", 
            f"Fold {fold_idx} Train"
        )
        
        # Process validation images
        self._process_images(
            fold_dir, 
            val_images, 
            "val", 
            f"Fold {fold_idx} Val"
        )
        
        # Copy test images
        self._copy_test_images(fold_dir)
        
        # Create data YAML
        self._create_data_yaml(fold_dir)
        
        return fold_dir
    
    def _process_images(
        self, 
        fold_dir: str, 
        images: np.ndarray, 
        split: str, 
        desc: str
    ):
        """Process and copy images with labels."""
        for img_name in tqdm(images, desc=desc):
            # Try both train and val source directories
            for src_dir in ["train", "val"]:
                src_img = os.path.join(self.config.img_dir, src_dir, img_name)
                
                if os.path.exists(src_img):
                    # Copy image
                    dst_img = os.path.join(fold_dir, "images", split, img_name)
                    shutil.copy(src_img, dst_img)
                    
                    # Create YOLO label
                    self._create_yolo_label(dst_img, img_name, fold_dir, split)
                    break
    
    def _create_yolo_label(
        self, 
        img_path: str, 
        img_name: str, 
        fold_dir: str, 
        split: str
    ):
        """Create YOLO format label file."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"  âš ï¸ Could not read image: {img_path}")
            return
            
        h, w, _ = img.shape
        
        label_file = os.path.join(
            fold_dir, 
            "labels", 
            split, 
            img_name.replace('.png', '.txt')
        )
        
        with open(label_file, 'w') as f:
            for _, row in self.image_to_annotations[img_name].iterrows():
                # Normalize coordinates
                xc = np.clip(row["x"] / w, 0, 1)
                yc = np.clip(row["y"] / h, 0, 1)
                bw = np.clip(row["width"] / w, 0, 1)
                bh = np.clip(row["height"] / h, 0, 1)
                
                # YOLO format: class x_center y_center width height
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    
    def _copy_test_images(self, fold_dir: str):
        """Copy test images to fold directory."""
        test_src = os.path.join(self.config.img_dir, "test")
        
        if os.path.exists(test_src):
            test_dst = os.path.join(fold_dir, "images", "test")
            test_images = [f for f in os.listdir(test_src) if f.endswith('.png')]
            
            print(f"  Copying {len(test_images)} test images...")
            for img_name in tqdm(test_images, desc="  Test images"):
                shutil.copy(
                    os.path.join(test_src, img_name),
                    os.path.join(test_dst, img_name)
                )
        else:
            print(f"  âš ï¸ Test directory not found: {test_src}")
    
    def _create_data_yaml(self, fold_dir: str):
        """Create YOLO data configuration YAML."""
        abs_fold_dir = os.path.abspath(fold_dir)
        data_yaml = {
            "path": abs_fold_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["cell"]
        }
        
        yaml_path = os.path.join(fold_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"  âœ“ Created {yaml_path}")


class DataPreparation:
    """Main data preparation orchestrator."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.loader = DatasetLoader(config)
        self.fold_creator = FoldCreator(config)
        
    def prepare(self) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """Prepare all folds and return fold directories."""
        print("="*80)
        print("RIVA DATA PREPARATION")
        print("="*80)
        print(f"Root Dir: {self.config.root_dir}")
        print(f"Work Dir: {self.config.work_dir}")
        print(f"K-Folds: {self.config.n_folds}")
        print("="*80)
        
        # Clean and create work directory
        if os.path.exists(self.config.work_dir):
            response = input(f"\nâš ï¸  {self.config.work_dir} exists. Remove it? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(self.config.work_dir)
                print(f"âœ“ Removed {self.config.work_dir}")
            else:
                print("âŒ Aborted")
                return [], {}
        
        os.makedirs(self.config.work_dir, exist_ok=True)
        
        # Load data
        combined_df, image_to_annotations = self.loader.load_data()
        
        # Create folds
        folds = self.fold_creator.create_folds(combined_df)
        
        # Prepare fold directories
        preparer = FoldPreparer(self.config, image_to_annotations)
        fold_dirs = []
        
        print("\nPreparing fold directories...")
        for fold in folds:
            print(f"\n{'='*80}")
            print(f"PREPARING FOLD {fold['fold']}")
            print(f"{'='*80}")
            
            fold_dir = preparer.prepare_fold(
                fold['fold'],
                fold['train_images'],
                fold['val_images']
            )
            fold_dirs.append(fold_dir)
        
        self._print_summary(fold_dirs)
        return fold_dirs, image_to_annotations
    
    def _print_summary(self, fold_dirs: List[str]):
        """Print preparation summary."""
        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE")
        print("="*80)
        print(f"âœ“ Prepared {len(fold_dirs)} folds")
        print(f"\nFold directories:")
        for fold_dir in fold_dirs:
            print(f"  - {fold_dir}")
        
        # Check first fold structure
        if fold_dirs:
            first_fold = fold_dirs[0]
            print(f"\nFirst fold structure ({first_fold}):")
            
            # Count files
            for split in ["train", "val", "test"]:
                img_dir = os.path.join(first_fold, "images", split)
                lbl_dir = os.path.join(first_fold, "labels", split)
                
                n_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.png')]) if os.path.exists(img_dir) else 0
                n_lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
                
                print(f"  {split:5s}: {n_imgs:4d} images, {n_lbls:4d} labels")
        
        print("\n" + "="*80)
        print("âœ“ Ready for training!")
        print(f"Run: python training.py --work_dir {self.config.work_dir}")
        print("="*80)


def verify_dataset_structure(root_dir: str) -> bool:
    """Verify the dataset has the expected structure."""
    print("\nVerifying dataset structure...")
    
    required_paths = {
        "train_csv": os.path.join(root_dir, "annotations", "train.csv"),
        "val_csv": os.path.join(root_dir, "annotations", "val.csv"),
        "train_images": os.path.join(root_dir, "images", "train"),
        "val_images": os.path.join(root_dir, "images", "val"),
        "test_images": os.path.join(root_dir, "images", "test"),
    }
    
    all_exist = True
    for name, path in required_paths.items():
        exists = os.path.exists(path)
        status = "âœ“" if exists else "âœ—"
        print(f"  {status} {name:15s}: {path}")
        if not exists and name != "test_images":  # test is optional during development
            all_exist = False
    
    return all_exist


def main():
    """Main data preparation entry point."""
    parser = argparse.ArgumentParser(description='RIVA Data Preparation - Standalone')
    parser.add_argument('--root_dir', type=str, default='riva-partb-dataset',
                       help='Root directory containing images/ and annotations/')
    parser.add_argument('--work_dir', type=str, default='yolov11x-swin',
                       help='Output directory for fold data')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify dataset structure, do not prepare')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Verify dataset structure
    if not verify_dataset_structure(args.root_dir):
        print("\nâŒ Dataset structure verification failed!")
        print("\nExpected structure:")
        print(f"{args.root_dir}/")
        print("â”œâ”€â”€ annotations/")
        print("â”‚   â”œâ”€â”€ train.csv")
        print("â”‚   â””â”€â”€ val.csv")
        print("â””â”€â”€ images/")
        print("    â”œâ”€â”€ train/")
        print("    â”œâ”€â”€ val/")
        print("    â””â”€â”€ test/")
        return
    
    print("âœ“ Dataset structure verified")
    
    if args.verify_only:
        print("\nVerification complete (--verify_only flag set)")
        return
    
    # Create config
    data_config = DataConfig(
        root_dir=args.root_dir,
        work_dir=args.work_dir,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    # Run preparation
    data_prep = DataPreparation(data_config)
    fold_dirs, image_to_annotations = data_prep.prepare()
    
    if not fold_dirs:
        print("\nâŒ Data preparation failed!")
        return
    
    # Save fold information
    fold_info = []
    for i, fold_dir in enumerate(fold_dirs):
        fold_info.append({
            'fold': i,
            'path': fold_dir,
            'data_yaml': os.path.join(fold_dir, 'data.yaml')
        })
    
    fold_info_df = pd.DataFrame(fold_info)
    fold_info_path = os.path.join(args.work_dir, "fold_info.csv")
    fold_info_df.to_csv(fold_info_path, index=False)
    print(f"\nâœ“ Fold information saved to {fold_info_path}")


if __name__ == "__main__":
    main()