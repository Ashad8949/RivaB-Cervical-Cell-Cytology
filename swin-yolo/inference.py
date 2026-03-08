"""
RIVA Part B - Standalone Inference Module
Can be run directly: python inference.py

Usage:
  python inference.py --test_dir path/to/test --models model1.pt model2.pt
  python inference.py --test_dir fold_0/images/test --work_dir yolov11x
"""

import os
import argparse
import glob
from typing import List, Tuple, Dict
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

from config import InferenceConfig


class MultiScaleTTAInference:
    """Multi-scale test-time augmentation inference."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
    def predict(
        self, 
        model: YOLO, 
        img_path: str
    ) -> Tuple[List, List]:
        """
        Run multi-scale TTA inference on a single image.
        
        Returns:
            boxes: List of normalized boxes [x1, y1, x2, y2]
            scores: List of confidence scores
        """
        all_boxes = []
        all_scores = []
        
        for scale in self.config.scales:
            # Original orientation
            boxes, scores = self._predict_at_scale(model, img_path, scale)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            
            # Horizontal flip augmentation
            boxes_flip, scores_flip = self._predict_flipped(model, img_path, scale)
            all_boxes.extend(boxes_flip)
            all_scores.extend(scores_flip)
        
        return all_boxes, all_scores
    
    def _predict_at_scale(
        self, 
        model: YOLO, 
        img_path: str, 
        scale: int
    ) -> Tuple[List, List]:
        """Predict at a single scale."""
        results = model.predict(
            source=img_path,
            imgsz=scale,
            conf=0.0001,  # Very low - WBF will filter
            iou=self.config.iou_threshold,
            augment=self.config.augment,
            max_det=self.config.max_detections,
            verbose=False
        )
        
        boxes = []
        scores = []
        
        if results[0].boxes is not None:
            boxes_xywhn = results[0].boxes.xywhn.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes_xywhn, confs):
                xc, yc, w, h = box
                # Convert to xyxy format
                boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
                scores.append(conf)
        
        return boxes, scores
    
    def _predict_flipped(
        self, 
        model: YOLO, 
        img_path: str, 
        scale: int
    ) -> Tuple[List, List]:
        """Predict on horizontally flipped image."""
        # Read and flip image
        img = cv2.imread(img_path)
        img_flip = cv2.flip(img, 1)
        
        # Save temporarily
        temp_path = "/tmp/temp_flip.png"
        cv2.imwrite(temp_path, img_flip)
        
        # Predict on flipped image
        results = model.predict(
            source=temp_path,
            imgsz=scale,
            conf=0.0001,
            iou=self.config.iou_threshold,
            augment=self.config.augment,
            max_det=self.config.max_detections,
            verbose=False
        )
        
        boxes = []
        scores = []
        
        if results[0].boxes is not None:
            boxes_xywhn = results[0].boxes.xywhn.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes_xywhn, confs):
                xc, yc, w, h = box
                # Flip x-coordinate back
                xc_orig = 1.0 - xc
                boxes.append([xc_orig - w/2, yc - h/2, xc_orig + w/2, yc + h/2])
                scores.append(conf)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return boxes, scores


class EnsemblePredictor:
    """Ensemble predictions from multiple models using WBF."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.tta_predictor = MultiScaleTTAInference(config)
        
    def predict_ensemble(
        self, 
        test_dir: str, 
        model_paths: List[str]
    ) -> List[Dict]:
        """
        Run ensemble inference on test set.
        
        Args:
            test_dir: Directory containing test images
            model_paths: List of paths to trained models
            
        Returns:
            List of prediction dictionaries
        """
        print(f"\n{'='*80}")
        print("RUNNING PROVEN INFERENCE STRATEGY")
        print(f"{'='*80}")
        print(f"Models: {len(model_paths)}")
        print(f"Scales: {self.config.scales}")
        print(f"WBF skip threshold: {self.config.wbf_skip_threshold}")
        print(f"{'='*80}\n")
        
        # Collect predictions from all models
        all_predictions = self._collect_predictions(test_dir, model_paths)
        
        # Fuse predictions with WBF
        final_predictions = self._fuse_predictions(all_predictions)
        
        return final_predictions
    
    def _collect_predictions(
        self, 
        test_dir: str, 
        model_paths: List[str]
    ) -> Dict:
        """Collect predictions from all models."""
        all_predictions = {}
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        
        if not test_images:
            print(f"âš ï¸ No PNG images found in {test_dir}")
            return all_predictions
        
        print(f"Found {len(test_images)} test images")
        
        for model_idx, model_path in enumerate(model_paths):
            print(f"\nModel {model_idx+1}/{len(model_paths)}: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"  âš ï¸ Model not found, skipping...")
                continue
                
            model = YOLO(model_path)
            
            for img_name in tqdm(test_images, desc="  Processing"):
                img_path = os.path.join(test_dir, img_name)
                boxes, scores = self.tta_predictor.predict(model, img_path)
                
                if img_name not in all_predictions:
                    all_predictions[img_name] = {'boxes': [], 'scores': []}
                
                all_predictions[img_name]['boxes'].extend(boxes)
                all_predictions[img_name]['scores'].extend(scores)
        
        return all_predictions
    
    def _fuse_predictions(self, all_predictions: Dict) -> List[Dict]:
        """Fuse predictions using Weighted Boxes Fusion."""
        final_predictions = []
        pred_id = 0
        
        print("\nApplying WBF fusion...")
        for img_name, pred_data in tqdm(all_predictions.items()):
            if not pred_data['boxes']:
                continue
            
            boxes = np.array(pred_data['boxes'])
            scores = np.array(pred_data['scores'])
            labels = np.zeros(len(boxes))
            
            # Apply WBF
            boxes_fused, scores_fused, _ = weighted_boxes_fusion(
                [boxes],
                [scores],
                [labels],
                weights=None,
                iou_thr=self.config.wbf_iou_threshold,
                skip_box_thr=self.config.wbf_skip_threshold,
                conf_type=self.config.wbf_conf_type
            )
            
            # Convert to predictions
            for box, conf in zip(boxes_fused, scores_fused):
                x1, y1, x2, y2 = box
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                final_predictions.append({
                    'id': pred_id,
                    'image_filename': img_name,
                    'x': float(x),
                    'y': float(y),
                    'width': float(w),
                    'height': float(h),
                    'conf': float(conf),
                    'class': 0
                })
                pred_id += 1
        
        return final_predictions


class SubmissionGenerator:
    """Generate submission files with multiple confidence thresholds."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
    def generate_submissions(
        self, 
        predictions: List[Dict], 
        test_dir: str,
        output_dir: str = "."
    ):
        """Generate submission files."""
        # Convert to DataFrame
        submission_df = pd.DataFrame(predictions)
        
        # Convert to absolute coordinates
        final_submission = self._to_absolute_coordinates(submission_df, test_dir)
        
        # Save main submission
        main_path = os.path.join(output_dir, "ultimate_submission.csv")
        final_submission.to_csv(main_path, index=False, float_format='%.6f')
        
        self._print_submission_stats(final_submission)
        
        # Generate threshold versions
        self._generate_threshold_versions(final_submission, output_dir)
    
    def _to_absolute_coordinates(
        self, 
        df: pd.DataFrame, 
        test_dir: str
    ) -> pd.DataFrame:
        """Convert normalized coordinates to absolute."""
        absolute = []
        
        for img_name in df['image_filename'].unique():
            img_preds = df[df['image_filename'] == img_name]
            img_path = os.path.join(test_dir, img_name)
            
            # Get image dimensions
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
            else:
                h, w = 1024, 1024
            
            # Convert predictions
            for _, row in img_preds.iterrows():
                absolute.append({
                    'id': row['id'],
                    'image_filename': img_name,
                    'class': 0,
                    'x': row['x'] * w,
                    'y': row['y'] * h,
                    'width': row['width'] * w,
                    'height': row['height'] * h,
                    'conf': row['conf']
                })
        
        return pd.DataFrame(absolute)
    
    def _generate_threshold_versions(
        self, 
        df: pd.DataFrame, 
        output_dir: str
    ):
        """Generate submissions with different confidence thresholds."""
        print("\n" + "="*80)
        print("GENERATING THRESHOLD VERSIONS")
        print("="*80)
        
        n_images = df['image_filename'].nunique()
        
        for thresh in self.config.final_thresholds:
            filtered = df[df['conf'] >= thresh].copy()
            filtered['id'] = range(len(filtered))
            
            output_path = os.path.join(
                output_dir, 
                f"ultimate_conf_{thresh:.2f}.csv"
            )
            filtered.to_csv(output_path, index=False, float_format='%.6f')
            
            if len(filtered) > 0:
                avg = len(filtered) / n_images
                print(f"  Threshold {thresh:.2f}: {len(filtered):>6,} predictions ({avg:>6.1f} per image)")
    
    def _print_submission_stats(self, df: pd.DataFrame):
        """Print submission statistics."""
        print("\n" + "="*80)
        print("SUBMISSION GENERATED")
        print("="*80)
        print(f"Total predictions: {len(df):,}")
        print(f"Unique images: {df['image_filename'].nunique()}")
        print(f"Avg per image: {len(df) / df['image_filename'].nunique():.1f}")
        print(f"Confidence: min={df['conf'].min():.4f}, max={df['conf'].max():.4f}, mean={df['conf'].mean():.4f}")
        
        print("\n" + "="*80)
        print("âœ“ COMPLETE - Ready for submission!")
        print("="*80)
        print("\nRecommended submission order:")
        print("  1. ultimate_conf_0.05.csv  (your proven threshold)")
        print("  2. ultimate_conf_0.08.csv")
        print("  3. ultimate_conf_0.10.csv")
        print("  4. ultimate_conf_0.15.csv")
        print("\nExpected: 0.55-0.70+ public score")
        print("="*80)


def find_model_paths(work_dir: str) -> List[str]:
    """Auto-detect trained model paths."""
    model_paths = []
    
    # Look for best.pt in runs directory
    pattern = os.path.join(work_dir, "runs", "fold_*", "weights", "best.pt")
    model_paths = sorted(glob.glob(pattern))
    
    return model_paths


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description='RIVA Inference - Standalone')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Model paths (space separated)')
    parser.add_argument('--work_dir', type=str, default=None,
                       help='Work directory to auto-find models (alternative to --models)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for submissions')
    parser.add_argument('--scales', type=int, nargs='+', default=[1024, 1280, 1536],
                       help='Image scales for multi-scale inference')
    parser.add_argument('--conf_threshold', type=float, default=0.05,
                       help='Confidence threshold')
    parser.add_argument('--wbf_skip', type=float, default=0.05,
                       help='WBF skip box threshold (CRITICAL!)')
    
    args = parser.parse_args()
    
    # Get model paths
    if args.models:
        model_paths = args.models
    elif args.work_dir:
        model_paths = find_model_paths(args.work_dir)
        if not model_paths:
            print(f"âŒ No models found in {args.work_dir}")
            print(f"Expected pattern: {args.work_dir}/runs/fold_*/weights/best.pt")
            return
    else:
        print("âŒ Must specify either --models or --work_dir")
        return
    
    # Verify test directory
    if not os.path.exists(args.test_dir):
        print(f"âŒ Test directory not found: {args.test_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config
    inference_config = InferenceConfig(
        scales=args.scales,
        conf_threshold=args.conf_threshold,
        wbf_skip_threshold=args.wbf_skip
    )
    
    print("="*80)
    print("RIVA STANDALONE INFERENCE")
    print("="*80)
    print(f"Test Dir: {args.test_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Models: {len(model_paths)}")
    for i, path in enumerate(model_paths):
        print(f"  {i+1}. {path}")
    print(f"Scales: {args.scales}")
    print(f"WBF Skip: {args.wbf_skip}")
    print("="*80)
    
    # Run inference
    predictor = EnsemblePredictor(inference_config)
    predictions = predictor.predict_ensemble(args.test_dir, model_paths)
    
    if not predictions:
        print("\nâŒ No predictions generated!")
        return
    
    # Generate submissions
    generator = SubmissionGenerator(inference_config)
    generator.generate_submissions(predictions, args.test_dir, args.output_dir)
    
    print(f"\nâœ“ Submissions saved to {args.output_dir}")


if __name__ == "__main__":
    main()