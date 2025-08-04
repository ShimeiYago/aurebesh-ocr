"""
Evaluate Aurebesh OCR pipeline on real images.
Computes detection mAP, recognition CER, and end-to-end metrics.
"""

import os
# Enable MPS fallback for operations not yet supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from datetime import datetime
from doctr.models.detection import db_mobilenet_v3_large
from doctr.models.recognition import crnn_mobilenet_v3_small
from torchvision import transforms

from utils import setup_logger, ensure_dir, get_timestamp, get_charset


class AurebeshEvaluator:
    def __init__(
        self,
        det_checkpoint: Path,
        rec_checkpoint: Path,
        images_dir: Path,
        output_dir: Path,
        device: str = 'mps'
    ):
        self.det_checkpoint = Path(det_checkpoint)
        self.rec_checkpoint = Path(rec_checkpoint)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        
        # Setup directories
        self.log_dir = ensure_dir(self.output_dir)
        
        # Setup logger
        self.logger = setup_logger('evaluate', self.log_dir, use_tensorboard=False)
        
        # Load models
        self.detector, self.recognizer = self._load_models()
        
        # Load data
        self.images, self.annotations = self._load_data()
        
        # Setup transforms
        self.det_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.rec_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load detector and recognizer models from checkpoints."""
        # Load detector
        det_ckpt = torch.load(self.det_checkpoint, map_location=self.device)
        detector = db_mobilenet_v3_large(pretrained=False)
        detector.load_state_dict(det_ckpt['model_state_dict'])
        
        # Create the same wrapper used in training
        class DBNetTrainingWrapper(torch.nn.Module):
            def __init__(self, dbnet_model):
                super().__init__()
                self.dbnet = dbnet_model
                
            def forward(self, x):
                outputs = self.dbnet(x, target=None, return_model_output=True)
                if isinstance(outputs, dict) and 'out_map' in outputs:
                    return outputs['out_map']
                else:
                    feat_maps = self.dbnet.feat_extractor(x)
                    feat_concat = self.dbnet.fpn(feat_maps)
                    logits = self.dbnet.prob_head(feat_concat)
                    return torch.sigmoid(logits)
        
        detector = DBNetTrainingWrapper(detector)
        detector.to(self.device)
        detector.eval()
        
        # Load recognizer
        rec_ckpt = torch.load(self.rec_checkpoint, map_location=self.device)
        charset = rec_ckpt.get('charset', get_charset())
        recognizer = crnn_mobilenet_v3_small(pretrained=False, vocab=charset)
        
        # Adapt output layer if needed
        vocab_size = len(charset)
        # Important: Model was trained with vocab_size + 1 outputs (CTC blank at index vocab_size)
        output_size = vocab_size + 1
        if hasattr(recognizer, 'classifier') and recognizer.classifier[-1].out_features != output_size:
            in_features = recognizer.classifier[-1].in_features
            recognizer.classifier[-1] = torch.nn.Linear(in_features, output_size)
        
        recognizer.load_state_dict(rec_ckpt['model_state_dict'])
        recognizer.to(self.device)
        recognizer.eval()
        
        self.charset = charset
        self.blank_idx = vocab_size  # CTC blank token is at vocab_size
        
        self.logger.info("Loaded detector and recognizer models")
        return detector, recognizer
    
    def _load_data(self) -> Tuple[List[Path], Dict]:
        """Load real images and annotations."""
        # Get image paths
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(self.images_dir.glob(f'images/{ext}'))
        
        # Load annotations if available
        ann_path = self.images_dir / 'annotations.json'
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = {}
            self.logger.warning("No annotations found, will only compute detection metrics")
        
        self.logger.info(f"Loaded {len(image_paths)} images")
        return sorted(image_paths), annotations
    
    def _detect_text(self, image: np.ndarray) -> List[Dict]:
        """Run text detection on image."""
        # Transform image
        img_tensor = self.det_transform(image).unsqueeze(0).to(self.device)
        
        # Run detection
        with torch.no_grad():
            pred_maps = self.detector(img_tensor)
        
        # Post-process predictions to get bounding boxes
        # The detector wrapper returns probability maps directly
        if isinstance(pred_maps, torch.Tensor):
            # If it's a tensor with shape [N, C, H, W], extract first sample and channel
            if pred_maps.ndim == 4:
                pred_map = pred_maps[0, 0].cpu().numpy()
            elif pred_maps.ndim == 3:
                pred_map = pred_maps[0].cpu().numpy()
            else:
                pred_map = pred_maps.cpu().numpy()
        else:
            raise ValueError(f"Unexpected detector output type: {type(pred_maps)}")
        
        # Threshold and find contours
        binary_map = (pred_map > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small boxes
            if w * h < 100:
                continue
            
            # Scale coordinates back to original image size
            h_scale = image.shape[0] / pred_map.shape[0]
            w_scale = image.shape[1] / pred_map.shape[1]
            
            bbox = [
                int(x * w_scale),
                int(y * h_scale),
                int((x + w) * w_scale),
                int((y + h) * h_scale)
            ]
            
            detections.append({
                'bbox': bbox,
                'score': float(pred_map[y:y+h, x:x+w].mean())
            })
        
        return detections
    
    def _recognize_text(self, image: np.ndarray, bbox: List[int]) -> str:
        """Run text recognition on cropped region."""
        # Crop image
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        
        # Transform
        crop_tensor = self.rec_transform(crop).unsqueeze(0).to(self.device)
        
        # Run recognition
        with torch.no_grad():
            # Pass dummy targets to avoid error
            outputs = self.recognizer(crop_tensor, target=[''], return_model_output=True)
        
        # Extract logits
        if isinstance(outputs, dict) and 'out_map' in outputs:
            logits = outputs['out_map']
        else:
            logits = outputs
        
        # Decode
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        # Decode text
        chars = []
        prev = None
        for idx in preds:
            if idx != self.blank_idx and idx != prev:  # blank_idx is vocab_size
                if idx < len(self.charset):
                    chars.append(self.charset[idx])
            prev = idx
        
        return ''.join(chars)
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate detection and recognition metrics."""
        # Detection metrics
        tp = 0
        fp = 0
        fn = 0
        
        # Match predictions to ground truth
        matched_gt = set()
        text_pairs = []
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > 0.5:  # IoU threshold
                tp += 1
                matched_gt.add(best_gt_idx)
                
                # Collect text pairs for CER calculation
                if 'text' in pred and 'text' in ground_truth[best_gt_idx]:
                    text_pairs.append((
                        pred['text'],
                        ground_truth[best_gt_idx]['text']
                    ))
            else:
                fp += 1
        
        fn = len(ground_truth) - len(matched_gt)
        
        # Calculate mAP@50
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate CER
        total_chars = 0
        total_errors = 0
        
        for pred_text, gt_text in text_pairs:
            total_chars += len(gt_text)
            # Simple CER calculation
            errors = sum(1 for a, b in zip(pred_text, gt_text) if a != b)
            errors += abs(len(pred_text) - len(gt_text))
            total_errors += errors
        
        cer = total_errors / max(total_chars, 1)
        
        # Harmonic mean of detection and recognition
        det_score = f1
        rec_score = 1 - cer
        hmean = 2 * det_score * rec_score / (det_score + rec_score) if (det_score + rec_score) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cer': cer,
            'hmean': hmean,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate(self):
        """Run evaluation on all images."""
        results = []
        
        self.logger.info("Starting evaluation...")
        
        for img_path in tqdm(self.images, desc="Evaluating"):
            # Load image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            detections = self._detect_text(image_rgb)
            
            # Run recognition on each detection
            for det in detections:
                text = self._recognize_text(image_rgb, det['bbox'])
                det['text'] = text
            
            # Get ground truth for this image
            gt_annotations = []
            if self.annotations:
                # Find matching annotations
                img_name = img_path.name
                for img_info in self.annotations.get('images', []):
                    if img_info.get('file_name') == img_name:
                        img_id = img_info['id']
                        # Get annotations for this image
                        for ann in self.annotations.get('annotations', []):
                            if ann['image_id'] == img_id:
                                x, y, w, h = ann['bbox']
                                gt_annotations.append({
                                    'bbox': [x, y, x + w, y + h],
                                    'text': ann.get('text', '')
                                })
                        break
            
            # Calculate metrics
            metrics = self._calculate_metrics(detections, gt_annotations)
            
            # Store results
            result = {
                'image': img_path.name,
                'num_detections': len(detections),
                'num_gt': len(gt_annotations),
                **metrics
            }
            results.append(result)
        
        # Calculate overall metrics
        overall_metrics = {
            'precision': np.mean([r['precision'] for r in results]),
            'recall': np.mean([r['recall'] for r in results]),
            'f1': np.mean([r['f1'] for r in results]),
            'cer': np.mean([r['cer'] for r in results if r['cer'] < float('inf')]),
            'hmean': np.mean([r['hmean'] for r in results])
        }
        
        # Save results to CSV
        timestamp = get_timestamp()
        csv_path = self.log_dir / f'eval_{timestamp}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Log overall metrics
        self.logger.info("Overall Metrics:")
        self.logger.info(f"  Precision: {overall_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {overall_metrics['recall']:.4f}")
        self.logger.info(f"  F1: {overall_metrics['f1']:.4f}")
        self.logger.info(f"  CER: {overall_metrics['cer']:.4f}")
        self.logger.info(f"  H-mean: {overall_metrics['hmean']:.4f}")
        
        self.logger.info(f"Results saved to {csv_path}")
        
        return overall_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Aurebesh OCR pipeline")
    parser.add_argument("--det_ckpt", type=Path, default="outputs/weights/det/best.pt", help="Detector checkpoint path")
    parser.add_argument("--rec_ckpt", type=Path, default="outputs/weights/rec/best.pt", help="Recognizer checkpoint path")
    parser.add_argument("--images", type=Path, default="data/synth/test", help="Images directory")
    parser.add_argument("--output_dir", type=Path, default="outputs/eval", help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    
    args = parser.parse_args()
    
    evaluator = AurebeshEvaluator(
        det_checkpoint=args.det_ckpt,
        rec_checkpoint=args.rec_ckpt,
        images_dir=args.images,
        output_dir=args.output_dir,
        device=args.device
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()