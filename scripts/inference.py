"""
Run inference on a single image using trained Aurebesh OCR models.
Outputs overlay visualization and JSON with detected text and bounding boxes.
"""

import os
# Enable MPS fallback for operations not yet supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from doctr.models.detection import db_mobilenet_v3_large
from doctr.models.recognition import crnn_mobilenet_v3_small
from torchvision import transforms

from utils import get_charset


class AurebeshOCR:
    def __init__(
        self,
        det_checkpoint: Path,
        rec_checkpoint: Path,
        device: str = 'mps'
    ):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        
        # Load models
        self.detector, self.recognizer = self._load_models(det_checkpoint, rec_checkpoint)
        
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
    
    def _load_models(self, det_path: Path, rec_path: Path) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load detector and recognizer models from checkpoints."""
        # Load detector
        det_ckpt = torch.load(det_path, map_location=self.device)
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
        rec_ckpt = torch.load(rec_path, map_location=self.device)
        self.charset = rec_ckpt.get('charset', get_charset())
        recognizer = crnn_mobilenet_v3_small(pretrained=False, vocab=self.charset)
        
        # Adapt output layer if needed
        vocab_size = len(self.charset)
        if hasattr(recognizer, 'classifier') and recognizer.classifier[-1].out_features != vocab_size:
            in_features = recognizer.classifier[-1].in_features
            recognizer.classifier[-1] = torch.nn.Linear(in_features, vocab_size)
        
        recognizer.load_state_dict(rec_ckpt['model_state_dict'])
        recognizer.to(self.device)
        recognizer.eval()
        
        return detector, recognizer
    
    def detect_text(self, image: np.ndarray) -> List[Dict]:
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
        
        # Sort by x-coordinate for reading order
        detections.sort(key=lambda d: d['bbox'][0])
        
        return detections
    
    def recognize_text(self, image: np.ndarray, bbox: List[int]) -> str:
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
            if idx != 0 and idx != prev:  # 0 is blank
                if idx < len(self.charset):
                    chars.append(self.charset[idx])
            prev = idx
        
        return ''.join(chars)
    
    def run_ocr(self, image_path: Path) -> Tuple[List[Dict], np.ndarray]:
        """Run complete OCR pipeline on image."""
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections = self.detect_text(image_rgb)
        
        # Run recognition on each detection
        results = []
        for det in detections:
            text = self.recognize_text(image_rgb, det['bbox'])
            result = {
                'text': text,
                'box': det['bbox'],
                'confidence': det['score']
            }
            results.append(result)
        
        # Create visualization
        vis_image = self._create_visualization(image, results)
        
        return results, vis_image
    
    def _create_visualization(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Create visualization with bounding boxes and text overlay."""
        vis_img = image.copy()
        
        # Define colors
        box_color = (0, 255, 0)  # Green
        text_color = (255, 255, 255)  # White
        bg_color = (0, 255, 0)  # Green background for text
        
        for result in results:
            x1, y1, x2, y2 = result['box']
            text = result['text']
            conf = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            
            # Prepare text label
            label = f"{text} ({conf:.2f})"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(vis_img, 
                         (x1, y1 - text_h - 4),
                         (x1 + text_w + 4, y1),
                         bg_color, -1)
            
            # Draw text
            cv2.putText(vis_img, label,
                       (x1 + 2, y1 - 2),
                       font, font_scale, text_color, thickness)
        
        return vis_img


def main():
    parser = argparse.ArgumentParser(description="Run Aurebesh OCR inference on an image")
    parser.add_argument("--img_path", type=Path, required=True, help="Path to input image")
    parser.add_argument("--det_ckpt", type=Path, default="outputs/weights/det/best.pt", help="Detector checkpoint path (default: outputs/weights/det/best.pt)")
    parser.add_argument("--rec_ckpt", type=Path, default="outputs/weights/rec/best.pt", help="Recognizer checkpoint path (default: outputs/weights/rec/best.pt)")
    parser.add_argument("--output_dir", type=Path, help="Output directory (optional)")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--save_json", action="store_true", help="Save results as JSON")
    parser.add_argument("--show", action="store_true", help="Display result image")
    
    args = parser.parse_args()
    
    # Resolve checkpoint paths relative to the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Convert relative paths to absolute paths
    if not args.det_ckpt.is_absolute():
        args.det_ckpt = project_root / args.det_ckpt
    if not args.rec_ckpt.is_absolute():
        args.rec_ckpt = project_root / args.rec_ckpt
    
    # Check if checkpoint files exist
    if not args.det_ckpt.exists():
        raise FileNotFoundError(f"Detection checkpoint not found: {args.det_ckpt}")
    if not args.rec_ckpt.exists():
        raise FileNotFoundError(f"Recognition checkpoint not found: {args.rec_ckpt}")
    
    # Initialize OCR
    ocr = AurebeshOCR(
        det_checkpoint=args.det_ckpt,
        rec_checkpoint=args.rec_ckpt,
        device=args.device
    )
    
    # Run OCR
    results, vis_image = ocr.run_ocr(args.img_path)
    
    # Print results to stdout
    print(f"\nDetected {len(results)} text regions:")
    for i, result in enumerate(results):
        print(f"{i+1}. '{result['text']}' at {result['box']} (conf: {result['confidence']:.3f})")
    
    # Save outputs if requested
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        base_name = args.img_path.stem
        vis_path = output_dir / f"{base_name}_result.png"
        cv2.imwrite(str(vis_path), vis_image)
        print(f"\nVisualization saved to: {vis_path}")
        
        # Save JSON if requested
        if args.save_json:
            json_path = output_dir / f"{base_name}_result.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {json_path}")
    
    # Display if requested
    if args.show:
        cv2.imshow("Aurebesh OCR Result", vis_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()