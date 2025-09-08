#!/usr/bin/env python3
"""
Generate detector-noise cropped dataset for recognizer training.

This script uses a trained detector to create a new recognizer dataset
with detector noise by cropping detected regions that match ground truth.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

from utils.inference_common import (
    load_config, pick_device, load_detector, run_detection_only,
    poly_iou, create_progress_bar
)
from utils.img_process import perspective_crop_polygon


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate detector-noise cropped dataset")
    
    parser.add_argument(
        "--det_path", 
        default="outputs/detector/weights.pt",
        help="Path to trained detector weights (.pt file)"
    )
    
    parser.add_argument(
        "--input", 
        default="data/synth/train",
        help="Dataset root containing images/ and labels.json"
    )
    
    parser.add_argument(
        "--config",
        default="configs/post_process.yaml", 
        help="Path to post-processing configuration file"
    )
    
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detected and ground truth polygons"
    )
    
    parser.add_argument(
        "--output_suffix",
        default="det-cropped",
        help="Output directory suffix (will be {input}/{suffix}/)"
    )
    
    return parser.parse_args()


def load_ground_truth_labels(labels_path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth labels from JSON file."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def match_detections_to_gt(
    detected_polygons: List[np.ndarray], 
    gt_polygons: List[List[List[int]]], 
    iou_threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """
    Match detected polygons to ground truth polygons based on IoU.
    
    Args:
        detected_polygons: List of detected polygon arrays, shape (4, 2)
        gt_polygons: List of ground truth polygons [[x,y], [x,y], ...]
        iou_threshold: Minimum IoU for matching
        
    Returns:
        List of (det_idx, gt_idx) tuples for matched pairs
    """
    matches = []
    used_gt_indices = set()
    
    for det_idx, det_poly in enumerate(detected_polygons):
        best_iou = 0.0
        best_gt_idx = -1
        
        # Convert detected polygon to list format for IoU calculation
        det_poly_list = det_poly.tolist()
        
        for gt_idx, gt_poly in enumerate(gt_polygons):
            if gt_idx in used_gt_indices:
                continue
                
            iou = poly_iou(det_poly_list, gt_poly)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append((det_idx, best_gt_idx))
            used_gt_indices.add(best_gt_idx)
    
    return matches


def create_cropped_dataset(args):
    """Main function to create detector-noise cropped dataset."""
    # Setup paths
    input_dir = Path(args.input)
    images_dir = input_dir / "images"
    labels_path = input_dir / "labels.json"
    
    output_dir = input_dir / args.output_suffix
    output_images_dir = output_dir / "images"
    output_labels_path = output_dir / "labels.json"
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    output_images_dir.mkdir(exist_ok=True)
    
    # Load configuration and model
    print("Loading configuration and detector model...")
    cfg = load_config(args.config)
    device = pick_device()
    detector = load_detector(args.det_path, cfg, device)
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    gt_labels = load_ground_truth_labels(str(labels_path))
    
    # Process images
    print("Processing images...")
    output_labels = {}
    crop_counter = 0
    
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    for img_filename in create_progress_bar(image_files, desc="Processing images"):
        try:
            # Skip if not in ground truth
            if img_filename not in gt_labels:
                print(f"Warning: {img_filename} not found in labels.json, skipping")
                continue
            
            img_path = images_dir / img_filename
            
            # Run detection
            detected_polygons_list = run_detection_only(detector, str(img_path))
            
            if not detected_polygons_list or len(detected_polygons_list) == 0:
                print(f"Warning: No detections for {img_filename}")
                continue
                
            detected_polygons = detected_polygons_list[0]  # First page
            
            # Get ground truth for this image
            gt_data = gt_labels[img_filename]
            gt_polygons = gt_data["polygons"]
            gt_texts = gt_data["texts"]
            
            # Match detections to ground truth
            matches = match_detections_to_gt(
                detected_polygons, gt_polygons, args.iou_threshold
            )
            
            if not matches:
                continue
            
            # Load original image
            original_image = Image.open(img_path)
            
            # Process matched detections
            for det_idx, gt_idx in matches:
                det_polygon = detected_polygons[det_idx]
                gt_text = gt_texts[gt_idx]
                
                # Convert to list format for perspective_crop_polygon
                polygon_list = det_polygon.tolist()
                
                try:
                    # Crop using detected polygon (with noise)
                    cropped_image = perspective_crop_polygon(original_image, polygon_list)
                    
                    # Save cropped image
                    crop_filename = f"crop_{crop_counter:06d}.jpg"
                    crop_path = output_images_dir / crop_filename
                    cropped_image.save(crop_path, "JPEG", quality=95)
                    
                    # Add to output labels (simple format like cropped dataset)
                    output_labels[crop_filename] = gt_text
                    
                    crop_counter += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to crop detection from {img_filename}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            continue
    
    # Save output labels
    print(f"Saving labels to {output_labels_path}...")
    with open(output_labels_path, 'w') as f:
        json.dump(output_labels, f, indent=2)
    
    print(f"Dataset creation completed!")
    print(f"- Total cropped images: {crop_counter}")
    print(f"- Output directory: {output_dir}")
    print(f"- Images saved to: {output_images_dir}")
    print(f"- Labels saved to: {output_labels_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    images_dir = input_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    labels_path = input_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    if not Path(args.det_path).exists():
        raise FileNotFoundError(f"Detector weights not found: {args.det_path}")
    
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Run dataset creation
    create_cropped_dataset(args)


if __name__ == "__main__":
    main()
