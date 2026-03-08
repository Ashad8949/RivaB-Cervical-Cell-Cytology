#!/usr/bin/env python
"""
Inference script for RIVA Cell Detection

Usage:
    python scripts/inference.py --checkpoint checkpoints/best.pth --test-dir ./test
    python scripts/inference.py --checkpoint checkpoints/best.pth --test-dir ./test --tta
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
import cv2
import json

from utils.io_utils import load_checkpoint, get_image_paths
from utils.visualization import draw_boxes, visualize_predictions
from inference.tta import TestTimeAugmentation, TTAConfig
from inference.postprocessing import PostProcessor
from models.hybrid_models import HybridCellDetector


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    model = HybridCellDetector(
        cnn_backbone_name=config.get('cnn_backbone', 'convnextv2_base'),
        transformer_backbone_name=config.get('transformer_backbone', 'path_dino'),
        detection_head_name=config.get('detection_head', 'dino_detr'),
        num_classes=config.get('num_classes', 1),
        num_queries=config.get('num_queries', 300),
        img_size=config.get('image_size', 1024),
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model, config


def run_inference(
    model,
    image_paths: list,
    device: str,
    use_tta: bool = False,
    tta_scales: list = None,
    confidence_threshold: float = 0.1,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
    save_visualizations: bool = False,
    output_dir: str = None
):
    """Run inference on images"""
    
    # TTA configuration
    if use_tta:
        tta_config = TTAConfig(
            scales=tta_scales or [896, 1024, 1152],
            flips=['none', 'horizontal', 'vertical'],
            rotations=[0, 90, 180, 270],
            merge_method='weighted_box_fusion',
            confidence_threshold=0.05,
            max_detections=max_detections * 2
        )
        tta_engine = TestTimeAugmentation(model, tta_config, device)
    else:
        tta_engine = None
    
    # Post-processor
    post_processor = PostProcessor(
        nms_threshold=nms_threshold,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections
    )
    
    # Results
    all_predictions = []
    all_filenames = []
    all_image_sizes = []
    
    for img_path in tqdm(image_paths, desc="Running inference"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Inference
        if tta_engine:
            predictions = tta_engine.predict(image_rgb, (h, w))
        else:
            # Simple inference without TTA
            predictions = single_inference(model, image_rgb, device)
        
        # Post-process
        predictions = post_processor(predictions, (h, w))
        
        all_predictions.append(predictions)
        all_filenames.append(img_path.name)
        all_image_sizes.append((h, w))
        
        # Save visualization
        if save_visualizations and output_dir:
            vis_dir = Path(output_dir) / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            vis_image = draw_boxes(
                image.copy(),
                predictions['boxes'],
                predictions['scores'],
                box_format='cxcywh',
                color=(0, 255, 0),
                thickness=2
            )
            
            cv2.imwrite(str(vis_dir / img_path.name), vis_image)
    
    return all_predictions, all_filenames, all_image_sizes


def single_inference(model, image: np.ndarray, device: str) -> dict:
    """Single image inference without TTA"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Preprocess
    h, w = image.shape[:2]
    target_size = 1024
    
    transform = A.Compose([
        A.LongestMaxSize(max_size=target_size),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Post-process outputs
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
    parser = argparse.ArgumentParser(description='RIVA Cell Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--test-dir', type=str, required=True, help='Test images directory')
    parser.add_argument('--output-dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--tta-scales', type=int, nargs='+', default=[896, 1024, 1152], help='TTA scales')
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-detections', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    parser.add_argument('--save-json', action='store_true', help='Save predictions as JSON')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Get image paths
    image_paths = get_image_paths(args.test_dir)
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    predictions, filenames, image_sizes = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        use_tta=args.tta,
        tta_scales=args.tta_scales,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        save_visualizations=args.save_vis,
        output_dir=str(output_dir)
    )
    
    # Summary
    total_detections = sum(len(p['boxes']) for p in predictions)
    print(f"\nInference completed:")
    print(f"  Total images: {len(predictions)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/image: {total_detections / max(1, len(predictions)):.1f}")
    
    # Save predictions as JSON
    if args.save_json:
        json_output = []
        for pred, filename, img_size in zip(predictions, filenames, image_sizes):
            h, w = img_size
            for i in range(len(pred['boxes'])):
                box = pred['boxes'][i]
                # Convert from normalized cxcywh to absolute xywh
                cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                x, y = cx - bw/2, cy - bh/2
                
                json_output.append({
                    'image_filename': filename,
                    'class': int(pred['labels'][i]),
                    'x': float(x),
                    'y': float(y),
                    'width': float(bw),
                    'height': float(bh),
                    'confidence': float(pred['scores'][i])
                })
        
        json_path = output_dir / 'predictions.json'
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Predictions saved to: {json_path}")
    
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
