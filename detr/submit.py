#!/usr/bin/env python
"""
Submission generation script for RIVA Cell Detection

Usage:
    python scripts/submit.py --checkpoint checkpoints/best.pth --test-dir ./test --output submission.csv
    python scripts/submit.py --ensemble-config ensemble.yaml --test-dir ./test --output submission.csv
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import cv2

from utils.io_utils import load_config, get_image_paths, generate_submission
from inference.tta import TestTimeAugmentation, TTAConfig
from inference.postprocessing import PostProcessor, weighted_box_fusion
from inference.ensemble_inference import ModelLoader, EnsembleInference
from models.hybrid_models import HybridCellDetector


def single_model_submission(args):
    """Generate submission using single model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    model = HybridCellDetector(
        num_classes=config.get('num_classes', 1),
        num_queries=config.get('num_queries', 300),
        img_size=config.get('image_size', 1024),
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # TTA
    if args.tta:
        tta_config = TTAConfig(
            scales=args.tta_scales,
            flips=['none', 'horizontal', 'vertical'],
            rotations=[0, 90, 180, 270],
            merge_method='weighted_box_fusion',
            confidence_threshold=0.05
        )
        inference_engine = TestTimeAugmentation(model, tta_config, device)
    else:
        inference_engine = None
    
    # Post-processor
    post_processor = PostProcessor(
        nms_threshold=args.nms_threshold,
        confidence_threshold=args.confidence,
        max_detections=args.max_detections
    )
    
    # Get images
    image_paths = get_image_paths(args.test_dir)
    print(f"Found {len(image_paths)} test images")
    
    # Run inference
    all_predictions = []
    all_filenames = []
    all_sizes = []
    
    for img_path in tqdm(image_paths, desc="Generating predictions"):
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        if inference_engine:
            predictions = inference_engine.predict(image_rgb, (h, w))
        else:
            predictions = simple_inference(model, image_rgb, device)
        
        predictions = post_processor(predictions, (h, w))
        
        all_predictions.append(predictions)
        all_filenames.append(img_path.name)
        all_sizes.append((h, w))
    
    # Generate submission
    generate_submission(
        predictions=all_predictions,
        image_filenames=all_filenames,
        output_path=args.output,
        image_sizes=all_sizes,
        confidence_threshold=args.confidence
    )


def ensemble_submission(args):
    """Generate submission using ensemble of models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load ensemble config
    config = load_config(args.ensemble_config)
    
    # Load models
    model_configs = config.get('models', [])
    if not model_configs:
        print("No models specified in ensemble config!")
        return
    
    print(f"Loading {len(model_configs)} models for ensemble...")
    model_loader = ModelLoader(model_configs, device)
    
    # Create ensemble
    tta_config = None
    if args.tta:
        tta_config = {
            'scales': args.tta_scales,
            'flips': ['none', 'horizontal'],
            'rotations': [0, 90],
            'merge_method': 'weighted_box_fusion'
        }
    
    ensemble = EnsembleInference(
        models=model_loader.get_models(),
        model_names=model_loader.get_model_names(),
        weights=config.get('weights', None),
        fusion_method=config.get('fusion_method', 'weighted_box_fusion'),
        device=device,
        use_tta=args.tta,
        tta_config=tta_config
    )
    
    # Post-processor
    post_processor = PostProcessor(
        nms_threshold=args.nms_threshold,
        confidence_threshold=args.confidence,
        max_detections=args.max_detections
    )
    
    # Get images
    image_paths = get_image_paths(args.test_dir)
    print(f"Found {len(image_paths)} test images")
    
    # Run inference
    all_predictions = []
    all_filenames = []
    all_sizes = []
    
    for img_path in tqdm(image_paths, desc="Ensemble inference"):
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        predictions = ensemble.predict(image_rgb, (h, w))
        predictions = post_processor(predictions, (h, w))
        
        all_predictions.append(predictions)
        all_filenames.append(img_path.name)
        all_sizes.append((h, w))
    
    # Generate submission
    generate_submission(
        predictions=all_predictions,
        image_filenames=all_filenames,
        output_path=args.output,
        image_sizes=all_sizes,
        confidence_threshold=args.confidence
    )


def simple_inference(model, image: np.ndarray, device: str) -> dict:
    """Simple inference without TTA"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    h, w = image.shape[:2]
    target_size = 1024
    
    transform = A.Compose([
        A.LongestMaxSize(max_size=target_size),
        A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    logits = outputs['pred_logits'].softmax(-1)
    boxes = outputs['pred_boxes']
    
    scores, labels = logits[0][:, :-1].max(dim=-1)
    keep = scores > 0.05
    
    return {
        'boxes': boxes[0][keep].cpu().numpy(),
        'scores': scores[keep].cpu().numpy(),
        'labels': labels[keep].cpu().numpy()
    }


def main():
    parser = argparse.ArgumentParser(description='Generate RIVA Submission')
    
    # Model options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--checkpoint', type=str, help='Single model checkpoint')
    model_group.add_argument('--ensemble-config', type=str, help='Ensemble config YAML')
    
    # Required
    parser.add_argument('--test-dir', type=str, required=True, help='Test images directory')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV path')
    
    # TTA options
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--tta-scales', type=int, nargs='+', default=[896, 1024, 1152], help='TTA scales')
    
    # Post-processing
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--max-detections', type=int, default=300, help='Max detections per image')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        single_model_submission(args)
    else:
        ensemble_submission(args)
    
    print(f"\nSubmission saved to: {args.output}")


if __name__ == '__main__':
    main()
