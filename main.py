#!/usr/bin/env python
"""
RIVA Track B Cell Detection - Main Entry Point

Usage:
    python main.py train --config configs/training_config.yaml
    python main.py inference --checkpoint checkpoints/best.pth --test-dir ./test
    python main.py submit --checkpoint checkpoints/best.pth --test-dir ./test --output submission.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train(args):
    """Run training"""
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    
    from utils.io_utils import load_config, setup_experiment_dir
    from utils.logging import get_logger, WandbLogger
    from data.dataset import RIVADataset, collate_fn
    from data.augmentations import TrainAugmentations, ValAugmentations
    from models.hybrid_models import HybridCellDetector
    from models.losses import DETRLoss, CellDetectionLoss
    from training.trainer import Trainer
    
    # Load config
    config = load_config(args.config)
    
    # Setup experiment directory
    exp_name = config.get('experiment_name', 'riva_cell_detection')
    exp_dirs = setup_experiment_dir(config.get('output_dir', './experiments'), exp_name)
    
    # Logger
    logger = get_logger('train', log_dir=str(exp_dirs['logs']))
    logger.info(f"Starting training: {exp_name}")
    logger.info(f"Config: {args.config}")
    
    # Wandb logger (optional)
    wandb_logger = None
    if config.get('use_wandb', False):
        wandb_logger = WandbLogger(
            project=config.get('wandb_project', 'riva-cell-detection'),
            name=exp_name,
            config=config
        )
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df = pd.read_csv(config['train_csv'])
    val_df = pd.read_csv(config['val_csv']) if 'val_csv' in config else None
    
    # Create datasets
    train_transform = TrainAugmentations(
        image_size=config.get('image_size', 1024),
        strong_aug=config.get('strong_augmentation', True),
        use_medical_aug=config.get('medical_augmentation', True)
    )
    
    val_transform = ValAugmentations(image_size=config.get('image_size', 1024))
    
    train_dataset = RIVADataset(
        train_df,
        config['image_dir'],
        transforms=train_transform.transform,
        is_train=True,
        image_size=config.get('image_size', 1024)
    )
    
    if val_df is not None:
        val_dataset = RIVADataset(
            val_df,
            config['image_dir'],
            transforms=val_transform.transform,
            is_train=False,
            image_size=config.get('image_size', 1024)
        )
    else:
        # Use train/val split
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=config.get('val_split', 0.1),
            random_state=42
        )
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    # Create data loaders
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
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = HybridCellDetector(
        cnn_backbone_name=config.get('cnn_backbone', 'convnextv2_base'),
        transformer_backbone_name=config.get('transformer_backbone', 'path_dino'),
        detection_head_name=config.get('detection_head', 'dino_detr'),
        num_classes=config.get('num_classes', 1),
        num_queries=config.get('num_queries', 300),
        img_size=config.get('image_size', 1024),
        pretrained=config.get('pretrained', True)
    )
    
    logger.info(f"Model created: {type(model).__name__}")
    
    # Loss function
    base_loss = DETRLoss(
        num_classes=config.get('num_classes', 1),
        weight_dict=config.get('loss_weights', None)
    )
    
    criterion = CellDetectionLoss(
        base_loss=base_loss,
        size_weight=config.get('size_loss_weight', 0.1),
        overlap_weight=config.get('overlap_loss_weight', 0.1)
    )
    
    # Optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config.get('lr', 1e-4)},
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': config.get('backbone_lr', 1e-5)}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Scheduler (with linear warmup)
    warmup_epochs = config.get('warmup_epochs', 3)
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('scheduler_T0', 10),
        T_mult=config.get('scheduler_T_mult', 2),
        eta_min=config.get('min_lr', 1e-7)
    )
    
    # Wrap with warmup: linearly ramp LR from 0 to base_lr over warmup_epochs
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine_scheduler
    
    # Trainer config
    trainer_config = {
        'mixed_precision': config.get('mixed_precision', True),
        'accumulation_steps': config.get('accumulation_steps', 1),
        'grad_clip_norm': config.get('grad_clip_norm', 0.1),
        'save_dir': str(exp_dirs['checkpoints']),
        'log_interval': config.get('log_interval', 10),
        'eval_every': config.get('eval_every', 1),
        'save_every': config.get('save_every', 5),
        'scheduler_type': 'epoch',
        'confidence_threshold': config.get('confidence_threshold', 0.1)
    }
    
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
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from: {args.resume}")
    
    # Train
    epochs = config.get('epochs', 100)
    history = trainer.fit(epochs)
    
    logger.info("Training completed!")
    logger.info(f"Best mAP: {trainer.best_metric:.4f}")
    
    # Cleanup
    if wandb_logger:
        wandb_logger.finish()
    
    return history


def inference(args):
    """Run inference on test images"""
    import torch
    import numpy as np
    import cv2
    from tqdm import tqdm
    
    from utils.io_utils import load_checkpoint, get_image_paths
    from inference.tta import TestTimeAugmentation, TTAConfig
    from inference.postprocessing import PostProcessor
    from models.hybrid_models import HybridCellDetector
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    model = HybridCellDetector(
        num_classes=config.get('num_classes', 1),
        num_queries=config.get('num_queries', 300),
        img_size=config.get('image_size', 1024),
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # TTA
    tta_config = TTAConfig(
        scales=[896, 1024, 1152] if args.tta else [1024],
        flips=['none', 'horizontal'] if args.tta else ['none'],
        rotations=[0] if not args.tta else [0, 90, 180, 270],
        merge_method='weighted_box_fusion',
        confidence_threshold=args.confidence
    )
    
    tta_engine = TestTimeAugmentation(model, tta_config, device)
    
    # Post-processor
    post_processor = PostProcessor(
        nms_threshold=args.nms_threshold,
        confidence_threshold=args.confidence,
        max_detections=args.max_detections
    )
    
    # Get test images
    image_paths = get_image_paths(args.test_dir)
    print(f"Found {len(image_paths)} test images")
    
    # Run inference
    all_predictions = []
    all_filenames = []
    
    for img_path in tqdm(image_paths, desc="Inference"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        predictions = tta_engine.predict(image)
        
        # Post-process
        predictions = post_processor(predictions, image.shape[:2])
        
        all_predictions.append(predictions)
        all_filenames.append(img_path.name)
    
    print(f"Inference completed. Total detections: {sum(len(p['boxes']) for p in all_predictions)}")
    
    return all_predictions, all_filenames


def submit(args):
    """Generate submission file"""
    from utils.io_utils import generate_submission
    
    # Run inference
    predictions, filenames = inference(args)
    
    # Get image sizes
    import cv2
    image_sizes = []
    for filename in filenames:
        img_path = Path(args.test_dir) / filename
        img = cv2.imread(str(img_path))
        image_sizes.append(img.shape[:2])
    
    # Generate submission
    generate_submission(
        predictions=predictions,
        image_filenames=filenames,
        output_path=args.output,
        image_sizes=image_sizes,
        confidence_threshold=args.confidence
    )
    
    print(f"Submission saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='RIVA Cell Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    train_parser.add_argument('--debug', action='store_true', help='Debug mode (smaller dataset)')
    
    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    infer_parser.add_argument('--test-dir', type=str, required=True, help='Test images directory')
    infer_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    infer_parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    infer_parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS threshold')
    infer_parser.add_argument('--max-detections', type=int, default=300, help='Max detections')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Generate submission')
    submit_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    submit_parser.add_argument('--test-dir', type=str, required=True, help='Test images directory')
    submit_parser.add_argument('--output', type=str, default='submission.csv', help='Output file')
    submit_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    submit_parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    submit_parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS threshold')
    submit_parser.add_argument('--max-detections', type=int, default=300, help='Max detections')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'inference':
        inference(args)
    elif args.command == 'submit':
        submit(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
