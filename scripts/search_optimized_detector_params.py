#!/usr/bin/env python3
"""
Grid search script for detector hyperparameter optimization.
Finds the best bin_thresh and unclip_ratio values for the trained detector model.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Any
import numpy as np
from itertools import product

# Import from utils
from utils.inference_common import (
    load_config, pick_device, load_detector, run_detection_only,
    list_images, poly_iou, create_progress_bar
)


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Grid search for detector hyperparameters")
    ap.add_argument("--det_path", default="outputs/detector/weights.pt",
                    help=".pt path for detector")
    ap.add_argument("--input", default="data/synth/test",
                    help="dataset root containing images/ and labels.json")
    return ap.parse_args()


def load_ground_truth(dataset_root: str) -> Dict[str, List[List[List[int]]]]:
    """
    Load ground truth annotations from labels.json.
    
    Returns:
        Dict mapping image filename to list of polygons
        Each polygon is a list of [x, y] coordinates
    """
    labels_path = os.path.join(dataset_root, "labels.json")
    with open(labels_path, "r") as f:
        data = json.load(f)
    
    gt_dict = {}
    # Handle both formats: list of dicts and dict with image keys
    if isinstance(data, list):
        # Format: [{"image": "name.jpg", "annotations": [...]}]
        for item in data:
            img_name = item["image"]
            polygons = []
            for ann in item.get("annotations", []):
                if "polygon" in ann:
                    polygons.append(ann["polygon"])
            gt_dict[img_name] = polygons
    elif isinstance(data, dict):
        # Format: {"img_name.jpg": {"polygons": [...]}}
        for img_name, annotations in data.items():
            polygons = annotations.get("polygons", [])
            gt_dict[img_name] = polygons
    
    return gt_dict


def calculate_metrics(pred_polygons: List[List[List[int]]], 
                     gt_polygons: List[List[List[int]]], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection metrics.
    
    Args:
        pred_polygons: List of predicted polygons
        gt_polygons: List of ground truth polygons  
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dict containing precision, recall, f1, mean_iou, and concatenation_rate
    """
    if len(pred_polygons) == 0 and len(gt_polygons) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0, 
            "f1": 1.0,
            "mean_iou": 1.0,
            "concatenation_rate": 0.0
        }
    
    if len(pred_polygons) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "concatenation_rate": 0.0
        }
    
    if len(gt_polygons) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0 if len(pred_polygons) > 0 else 1.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "concatenation_rate": 1.0  # All predictions are false positives (concatenations)
        }
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_polygons), len(gt_polygons)))
    for i, pred_poly in enumerate(pred_polygons):
        for j, gt_poly in enumerate(gt_polygons):
            iou_matrix[i, j] = poly_iou(pred_poly, gt_poly)
    
    # Find best matches
    matched_preds = set()
    matched_gts = set()
    match_ious = []
    
    # Greedy matching: highest IoU first
    while True:
        max_iou = 0.0
        max_pos = None
        
        for i in range(len(pred_polygons)):
            if i in matched_preds:
                continue
            for j in range(len(gt_polygons)):
                if j in matched_gts:
                    continue
                if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= iou_threshold:
                    max_iou = iou_matrix[i, j]
                    max_pos = (i, j)
        
        if max_pos is None:
            break
            
        matched_preds.add(max_pos[0])
        matched_gts.add(max_pos[1])
        match_ious.append(max_iou)
    
    # Calculate metrics
    tp = len(match_ious)
    fp = len(pred_polygons) - tp
    fn = len(gt_polygons) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_iou = np.mean(match_ious) if match_ious else 0.0
    
    # Concatenation rate: ratio of predictions that don't match any GT
    concatenation_rate = fp / len(pred_polygons) if len(pred_polygons) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "concatenation_rate": concatenation_rate
    }


def evaluate_params(det_path: str, image_paths: List[str], 
                   gt_dict: Dict[str, List[List[List[int]]]], 
                   bin_thresh: float, unclip_ratio: float) -> Dict[str, float]:
    """
    Evaluate detector with given hyperparameters.
    
    Returns:
        Dict containing aggregated metrics
    """
    # Load base config from file
    cfg = load_config("configs/post_process.yaml")
    
    # Override detector parameters with test values
    cfg["detector"]["bin_thresh"] = bin_thresh
    cfg["detector"]["unclip_ratio"] = unclip_ratio
    cfg["detector"]["box_thresh"] = 0.1  # Fixed
    
    # Load detector with new parameters
    device = pick_device()
    det = load_detector(det_path, cfg, device)
    
    all_metrics = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Run detection
        try:
            pred_polygons_list = run_detection_only(det, img_path)
            # Assume single page
            pred_polygons = pred_polygons_list[0].tolist() if len(pred_polygons_list) > 0 and pred_polygons_list[0].size > 0 else []
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            pred_polygons = []
        
        # Get ground truth
        gt_polygons = gt_dict.get(img_name, [])
        
        # Calculate metrics for this image
        metrics = calculate_metrics(pred_polygons, gt_polygons)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    if not all_metrics:
        return {
            "f1": 0.0,
            "mean_iou": 0.0,
            "concatenation_rate": 1.0,
            "precision": 0.0,
            "recall": 0.0
        }
    
    aggregated = {}
    for key in all_metrics[0].keys():
        aggregated[key] = np.mean([m[key] for m in all_metrics])
    
    return aggregated


def get_search_space(phase: int, best_params: Dict[str, float] = None) -> Tuple[List[float], List[float]]:
    """
    Get search space for given phase.
    
    Returns:
        Tuple of (unclip_ratio_values, bin_thresh_values)
    """
    if phase == 1:
        # Phase 1: Coarse search
        unclip_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        bin_threshes = [0.20, 0.25, 0.30, 0.35, 0.40]
        
    elif phase == 2:
        # Phase 2: Fine search around best parameters
        if best_params is None:
            raise ValueError("best_params required for phase 2")
        
        u_best = best_params["unclip_ratio"]
        b_best = best_params["bin_thresh"]
        
        # unclip_ratio: [u*-0.15, u*+0.15] with 0.02 step
        unclip_ratios = np.arange(max(0.5, u_best - 0.15), u_best + 0.15 + 0.001, 0.02).tolist()
        # bin_thresh: [b*-0.06, b*+0.06] with 0.01 step  
        bin_threshes = np.arange(max(0.05, b_best - 0.06), min(1.0, b_best + 0.06) + 0.001, 0.01).tolist()
        
    elif phase == 3:
        # Phase 3: Micro adjustment
        if best_params is None:
            raise ValueError("best_params required for phase 3")
            
        u_best = best_params["unclip_ratio"]
        b_best = best_params["bin_thresh"]
        
        # unclip_ratio: [u_best-0.06, u_best+0.06] with 0.01 step
        unclip_ratios = np.arange(max(0.5, u_best - 0.06), u_best + 0.06 + 0.001, 0.01).tolist()
        # bin_thresh: [b_best-0.03, b_best+0.03] with 0.005 step
        bin_threshes = np.arange(max(0.05, b_best - 0.03), min(1.0, b_best + 0.03) + 0.001, 0.005).tolist()
    
    else:
        raise ValueError(f"Invalid phase: {phase}")
    
    return unclip_ratios, bin_threshes


def run_phase(phase: int, det_path: str, image_paths: List[str], 
              gt_dict: Dict[str, List[List[List[int]]]], 
              best_params: Dict[str, float] = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Run a single optimization phase.
    
    Returns:
        Tuple of (best_result, all_results)
    """
    print(f"\n{'='*60}")
    print(f"Phase {phase}: {'Coarse search' if phase == 1 else 'Fine search' if phase == 2 else 'Micro adjustment'}")
    print(f"{'='*60}")
    
    # Get search space
    unclip_ratios, bin_threshes = get_search_space(phase, best_params)
    print(f"Search space: {len(unclip_ratios)} × {len(bin_threshes)} = {len(unclip_ratios) * len(bin_threshes)} combinations")
    print(f"unclip_ratio range: {min(unclip_ratios):.3f} - {max(unclip_ratios):.3f}")
    print(f"bin_thresh range: {min(bin_threshes):.3f} - {max(bin_threshes):.3f}")
    
    if best_params:
        print(f"Starting from best params: unclip_ratio={best_params['unclip_ratio']:.3f}, bin_thresh={best_params['bin_thresh']:.3f}")
    
    # Grid search
    results = []
    best_result = None
    best_score = None
    
    start_time = time.time()
    
    param_combinations = list(product(unclip_ratios, bin_threshes))
    progress_bar = create_progress_bar(param_combinations, desc=f"Phase {phase}")
    
    for unclip_ratio, bin_thresh in progress_bar:
        # Evaluate current parameters
        metrics = evaluate_params(
            det_path=det_path,
            image_paths=image_paths,
            gt_dict=gt_dict,
            bin_thresh=bin_thresh,
            unclip_ratio=unclip_ratio
        )
        
        # Store result
        result = {
            "unclip_ratio": float(unclip_ratio),
            "bin_thresh": float(bin_thresh),
            **metrics
        }
        results.append(result)
        
        # Update best result
        if best_result is None or compare_results(metrics, best_score) > 0:
            best_result = result.copy()
            best_score = metrics.copy()
        
        # Update progress bar description
        progress_bar.set_postfix({
            'Best F1': f'{best_score["f1"]:.4f}',
            'Best IoU': f'{best_score["mean_iou"]:.4f}',
            'Best Concat': f'{best_score["concatenation_rate"]:.4f}'
        })
    
    elapsed_time = time.time() - start_time
    
    # Sort results by performance
    results.sort(key=lambda x: (x["f1"], x["mean_iou"], -x["concatenation_rate"]), reverse=True)
    
    # Print results
    print(f"\nPhase {phase} completed in {elapsed_time:.2f} seconds")
    print(f"Evaluated {len(results)} parameter combinations")
    print(f"\nTop 5 results:")
    print(f"{'Rank':<4} {'unclip_ratio':<12} {'bin_thresh':<10} {'F1':<8} {'Mean IoU':<9} {'Concat Rate':<11} {'Precision':<9} {'Recall':<6}")
    print("-" * 80)
    
    for i, result in enumerate(results[:5]):
        print(f"{i+1:<4} {result['unclip_ratio']:<12.3f} {result['bin_thresh']:<10.3f} "
              f"{result['f1']:<8.4f} {result['mean_iou']:<9.4f} {result['concatenation_rate']:<11.4f} "
              f"{result['precision']:<9.4f} {result['recall']:<6.4f}")
    
    print(f"\nPhase {phase} best parameters:")
    print(f"unclip_ratio: {best_result['unclip_ratio']:.3f}")
    print(f"bin_thresh: {best_result['bin_thresh']:.3f}")
    print(f"F1 score: {best_result['f1']:.4f}")
    print(f"Mean IoU: {best_result['mean_iou']:.4f}")
    print(f"Concatenation rate: {best_result['concatenation_rate']:.4f}")
    
    return best_result, results


def compare_results(result1: Dict[str, float], result2: Dict[str, float]) -> int:
    """
    Compare two results according to the priority:
    1. F1@IoU=0.5 (higher is better)
    2. Mean IoU (higher is better)  
    3. Concatenation rate (lower is better)
    
    Returns:
        1 if result1 > result2, -1 if result1 < result2, 0 if equal
    """
    # Compare F1 score
    if result1["f1"] > result2["f1"]:
        return 1
    elif result1["f1"] < result2["f1"]:
        return -1
    
    # Compare Mean IoU
    if result1["mean_iou"] > result2["mean_iou"]:
        return 1
    elif result1["mean_iou"] < result2["mean_iou"]:
        return -1
    
    # Compare concatenation rate (lower is better)
    if result1["concatenation_rate"] < result2["concatenation_rate"]:
        return 1
    elif result1["concatenation_rate"] > result2["concatenation_rate"]:
        return -1
    
    return 0
    """
    Compare two results according to the priority:
    1. F1@IoU=0.5 (higher is better)
    2. Mean IoU (higher is better)  
    3. Concatenation rate (lower is better)
    
    Returns:
        1 if result1 > result2, -1 if result1 < result2, 0 if equal
    """
    # Compare F1 score
    if result1["f1"] > result2["f1"]:
        return 1
    elif result1["f1"] < result2["f1"]:
        return -1
    
    # Compare Mean IoU
    if result1["mean_iou"] > result2["mean_iou"]:
        return 1
    elif result1["mean_iou"] < result2["mean_iou"]:
        return -1
    
    # Compare concatenation rate (lower is better)
    if result1["concatenation_rate"] < result2["concatenation_rate"]:
        return 1
    elif result1["concatenation_rate"] > result2["concatenation_rate"]:
        return -1
    
    return 0


def main():
    args = parse_args()
    
    print("Starting detector hyperparameter optimization")
    print(f"Detector model: {args.det_path}")
    print(f"Dataset: {args.input}")
    
    # Load test images
    image_paths = list_images(args.input)
    print(f"Found {len(image_paths)} test images")
    
    # Load ground truth
    gt_dict = load_ground_truth(args.input)
    print(f"Loaded ground truth for {len(gt_dict)} images")
    
    # Phase 1: Coarse search
    phase1_best, phase1_results = run_phase(1, args.det_path, image_paths, gt_dict)
    
    # Phase 2: Fine search around phase 1 best
    phase2_best, phase2_results = run_phase(2, args.det_path, image_paths, gt_dict, 
                                           {"unclip_ratio": phase1_best["unclip_ratio"], 
                                            "bin_thresh": phase1_best["bin_thresh"]})
    
    # Phase 3: Micro adjustment around phase 2 best
    phase3_best, phase3_results = run_phase(3, args.det_path, image_paths, gt_dict,
                                           {"unclip_ratio": phase2_best["unclip_ratio"], 
                                            "bin_thresh": phase2_best["bin_thresh"]})
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nPhase 1 (Coarse) best:")
    print(f"  unclip_ratio: {phase1_best['unclip_ratio']:.3f}, bin_thresh: {phase1_best['bin_thresh']:.3f}")
    print(f"  F1: {phase1_best['f1']:.4f}, Mean IoU: {phase1_best['mean_iou']:.4f}, Concat Rate: {phase1_best['concatenation_rate']:.4f}")
    
    print(f"\nPhase 2 (Fine) best:")
    print(f"  unclip_ratio: {phase2_best['unclip_ratio']:.3f}, bin_thresh: {phase2_best['bin_thresh']:.3f}")
    print(f"  F1: {phase2_best['f1']:.4f}, Mean IoU: {phase2_best['mean_iou']:.4f}, Concat Rate: {phase2_best['concatenation_rate']:.4f}")
    
    print(f"\nPhase 3 (Micro) best:")
    print(f"  unclip_ratio: {phase3_best['unclip_ratio']:.3f}, bin_thresh: {phase3_best['bin_thresh']:.3f}")
    print(f"  F1: {phase3_best['f1']:.4f}, Mean IoU: {phase3_best['mean_iou']:.4f}, Concat Rate: {phase3_best['concatenation_rate']:.4f}")
    
    print(f"\nFINAL RECOMMENDED PARAMETERS:")
    print(f"unclip_ratio: {phase3_best['unclip_ratio']:.3f}")
    print(f"bin_thresh: {phase3_best['bin_thresh']:.3f}")
    
    # Performance comparison
    improvement_f1 = phase3_best['f1'] - phase1_best['f1']
    improvement_iou = phase3_best['mean_iou'] - phase1_best['mean_iou']
    improvement_concat = phase1_best['concatenation_rate'] - phase3_best['concatenation_rate']
    
    print(f"\nIMPROVEMENT FROM PHASE 1 TO PHASE 3:")
    print(f"F1 score: {improvement_f1:+.4f} ({'better' if improvement_f1 > 0 else 'worse' if improvement_f1 < 0 else 'same'})")
    print(f"Mean IoU: {improvement_iou:+.4f} ({'better' if improvement_iou > 0 else 'worse' if improvement_iou < 0 else 'same'})")
    print(f"Concat Rate: {improvement_concat:+.4f} ({'better' if improvement_concat > 0 else 'worse' if improvement_concat < 0 else 'same'})")

    print(f"\n It is recommended to update unclip_ratio: {phase3_best['unclip_ratio']:.3f} and bin_thresh: {phase3_best['bin_thresh']:.3f} in configs/post_process.yaml")


if __name__ == "__main__":
    main()
