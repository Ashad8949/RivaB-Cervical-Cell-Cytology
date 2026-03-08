#!/usr/bin/env python
"""
Training script for RIVA Cell Detection

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --fold 0
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.io_utils import load_config, setup_experiment_dir, save_config
from utils.logging import get_logger, WandbLogger, MetricTracker
from utils.visualization import plot_training_curves
from data.dataset import RIVADataset, collate_fn
from data.augmentations import TrainAugmentations, ValAugmentations
from models.hybrid_models import HybridCellDetector
from models.losses import DETRLoss, CellDetectionLoss
from training.trainer import Trainer


def create_folds(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Create stratified folds based on cell count per image"""
    # Count cells per image
    cell_counts = df.groupby('image_filename').size().reset_index(name='count')
    
    # Create bins for stratification
    cell_counts['count_bin'] = pd.cut(
        cell_counts['count'],
        bins=[0, 10, 50, 100, 500, float('inf')],
        labels=[0, 1, 2, 3, 4]
    )
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cell_counts['fold'] = -1
    
    for fold, (_, val_idx) in enumerate(skf.split(cell_counts, cell_counts['count_bin'])):
        cell_counts.loc[cell_counts.index[val_idx], 'fold'] = fold
    
    # Merge back
    df = df.merge(cell_counts[['image_filename', 'fold']], on='image_filename', how='left')
    
    return df


def train_fold(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
    exp_dirs: dict,
    device: str
):
    """Train single fold"""
    logger = get_logger(f'train_fold{fold}', log_dir=str(exp_dirs['logs']))
    logger.info(f"Training fold {fold}")
    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    
    # Wandb
    wandb_logger = None
    if config.get('use_wandb', False):
        wandb_logger = WandbLogger(
            project=config.get('wandb_project', 'riva-cell-detection'),
            name=f"{config.get('experiment_name', 'exp')}_fold{fold}",
            config=config,
            tags=[f'fold{fold}']
        )
    
    # Transforms
    train_transform = TrainAugmentations(
        image_size=config.get('image_size', 1024),
        strong_aug=config.get('strong_augmentation', True),
        use_medical_aug=config.get('medical_augmentation', True)
    )
    
    val_transform = ValAugmentations(image_size=config.get('image_size', 1024))
    
    # Datasets
    train_dataset = RIVADataset(
        train_df,
        config['image_dir'],
        transforms=train_transform.transform,
        is_train=True,
        image_size=config.get('image_size', 1024)
    )
    
    val_dataset = RIVADataset(
        val_df,
        config['image_dir'],
        transforms=val_transform.transform,
        is_train=False,
        image_size=config.get('image_size', 1024)
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model
    model = HybridCellDetector(
        cnn_backbone_name=config.get('cnn_backbone', 'convnextv2_base'),
        transformer_backbone_name=config.get('transformer_backbone', 'path_dino'),
        detection_head_name=config.get('detection_head', 'dino_detr'),
        num_classes=config.get('num_classes', 1),
        num_queries=config.get('num_queries', 300),
        img_size=config.get('image_size', 1024),
        pretrained=config.get('pretrained', True)
    )
    
    # Loss
    base_loss = DETRLoss(
        num_classes=config.get('num_classes', 1),
        weight_dict=config.get('loss_weights', None)
    )
    
    criterion = CellDetectionLoss(
        base_loss=base_loss,
        size_weight=config.get('size_loss_weight', 0.1),
        overlap_weight=config.get('overlap_loss_weight', 0.1)
    )
    
    # Optimizer with differential learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name.lower():
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.get('lr', 1e-4)},
        {'params': backbone_params, 'lr': config.get('backbone_lr', 1e-5)}
    ], weight_decay=config.get('weight_decay', 1e-4))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('scheduler_T0', 10),
        T_mult=config.get('scheduler_T_mult', 2),
        eta_min=config.get('min_lr', 1e-7)
    )
    
    # Trainer config
    trainer_config = {
        'mixed_precision': config.get('mixed_precision', True),
        'accumulation_steps': config.get('accumulation_steps', 1),
        'grad_clip_norm': config.get('grad_clip_norm', 0.1),
        'save_dir': str(exp_dirs['checkpoints'] / f'fold{fold}'),
        'log_interval': config.get('log_interval', 10),
        'eval_every': config.get('eval_every', 1),
        'save_every': config.get('save_every', 5),
        'scheduler_type': 'epoch',
        'confidence_threshold': config.get('confidence_threshold', 0.1)
    }
    
    os.makedirs(trainer_config['save_dir'], exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=trainer_config,
        logger=wandb_logger
    )
    
    # Train
    history = trainer.fit(config.get('epochs', 100))
    
    # Save training curves
    plot_training_curves(
        history,
        save_path=str(exp_dirs['visualizations'] / f'training_curves_fold{fold}.png'),
        show=False
    )
    
    logger.info(f"Fold {fold} completed. Best mAP: {trainer.best_metric:.4f}")
    
    if wandb_logger:
        wandb_logger.finish()
    
    return trainer.best_metric


def main():
    parser = argparse.ArgumentParser(description='Train RIVA Cell Detection')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--fold', type=int, default=None, help='Single fold to train (0-4)')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.debug:
        config['epochs'] = 2
        config['batch_size'] = 1
        print("Debug mode enabled")
    
    # Setup
    exp_name = config.get('experiment_name', 'riva_cell_detection')
    exp_dirs = setup_experiment_dir(config.get('output_dir', './experiments'), exp_name)
    
    # Save config
    save_config(config, str(exp_dirs['root'] / 'config.yaml'))
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(config['train_csv'])
    print(f"Total annotations: {len(df)}")
    
    # Create folds
    df = create_folds(df, n_folds=args.n_folds)
    
    # Train
    fold_scores = []
    
    folds_to_train = [args.fold] if args.fold is not None else range(args.n_folds)
    
    for fold in folds_to_train:
        train_df = df[df['fold'] != fold].copy()
        val_df = df[df['fold'] == fold].copy()
        
        score = train_fold(config, train_df, val_df, fold, exp_dirs, device)
        fold_scores.append(score)
    
    # Summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    for fold, score in enumerate(fold_scores):
        print(f"  Fold {fold}: mAP = {score:.4f}")
    print(f"  Mean mAP: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
