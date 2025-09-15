#!/usr/bin/env python3
"""
Grid search script for detector hyperparameter optimization with improved evaluation.

New evaluation method:
- Score sweeping for AP@IoU=0.5 using real objectness scores (box_thresh independent)  
- Fixed recall point evaluation (Recall=95%)
- Dataset weighting (Real: 0.8, Synthetic: 0.2)
- Safety metrics: Word split rate↓, GT coverage rate↑, Best IoU median↑

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
                    help="dataset of synthetic root containing images/ and labels.json")
    ap.add_argument("--input_real", default="data/real",
                    help="dataset of real root containing images/ and labels.json")
    ap.add_argument("--skip_phase3", action="store_true",
                    help="Skip phase 3 (micro adjustment) to save time")
    ap.add_argument("--real_weight", type=float, default=0.8,
                    help="Weight for real dataset in evaluation (0.0-1.0, default: 0.8)")
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


def calculate_ap_at_iou(detections_with_scores: List[Tuple[List[List[int]], float]], 
                       gt_polygons: List[List[List[int]]], 
                       iou_threshold: float = 0.5) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Calculate AP@IoU using score sweeping (box_thresh independent).
    
    Args:
        detections_with_scores: List of (polygon, confidence_score) tuples
        gt_polygons: List of ground truth polygons
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Tuple of (AP@IoU, precision_recall_curve)
    """
    if len(detections_with_scores) == 0:
        return 0.0, [(0.0, 0.0)]
    
    if len(gt_polygons) == 0:
        return 0.0, [(0.0, 1.0)]  # All detections are false positives
    
    # Sort detections by confidence score (descending)
    sorted_detections = sorted(detections_with_scores, key=lambda x: x[1], reverse=True)
    
    # Calculate IoU matrix between all detections and GT
    iou_matrix = np.zeros((len(sorted_detections), len(gt_polygons)))
    for i, (pred_poly, _) in enumerate(sorted_detections):
        for j, gt_poly in enumerate(gt_polygons):
            iou_matrix[i, j] = poly_iou(pred_poly, gt_poly)
    
    # Score sweeping to calculate precision-recall curve
    pr_curve = []
    matched_gts = set()
    
    for k in range(len(sorted_detections)):
        # Consider top k+1 detections
        current_detections = sorted_detections[:k+1]
        
        # Find matches for current detections
        temp_matched_gts = set()
        tp = 0
        
        for i, (pred_poly, _) in enumerate(current_detections):
            best_match = -1
            best_iou = 0.0
            
            for j in range(len(gt_polygons)):
                if j in temp_matched_gts:
                    continue
                if iou_matrix[i, j] >= iou_threshold and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_match = j
            
            if best_match >= 0:
                temp_matched_gts.add(best_match)
                tp += 1
        
        fp = len(current_detections) - tp
        fn = len(gt_polygons) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        pr_curve.append((precision, recall))
    
    # Calculate AP using trapezoidal rule
    if len(pr_curve) == 0:
        return 0.0, [(0.0, 0.0)]
    
    # Sort by recall for proper AP calculation
    pr_curve.sort(key=lambda x: x[1])
    
    # Add (0,1) and (1,0) points if needed
    if pr_curve[0][1] > 0:
        pr_curve.insert(0, (1.0, 0.0))
    if pr_curve[-1][1] < 1:
        pr_curve.append((0.0, 1.0))
    
    # Calculate AP
    ap = 0.0
    for i in range(1, len(pr_curve)):
        recall_diff = pr_curve[i][1] - pr_curve[i-1][1]
        avg_precision = (pr_curve[i][0] + pr_curve[i-1][0]) / 2.0
        ap += recall_diff * avg_precision
    
    return ap, pr_curve


def calculate_metrics_at_fixed_recall(detections_with_scores: List[Tuple[List[List[int]], float]], 
                                     gt_polygons: List[List[List[int]]], 
                                     target_recall: float = 0.95,
                                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics at fixed recall point (e.g., 95%).
    
    Returns:
        Dict containing precision, actual_recall, f1, mean_iou, etc. at target recall
    """
    if len(detections_with_scores) == 0 or len(gt_polygons) == 0:
        return {
            "precision_at_recall": 0.0,
            "actual_recall": 0.0,
            "f1_at_recall": 0.0,
            "mean_iou_at_recall": 0.0,
            "threshold_for_recall": 0.0
        }
    
    # Sort detections by confidence score (descending)
    sorted_detections = sorted(detections_with_scores, key=lambda x: x[1], reverse=True)
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(sorted_detections), len(gt_polygons)))
    for i, (pred_poly, _) in enumerate(sorted_detections):
        for j, gt_poly in enumerate(gt_polygons):
            iou_matrix[i, j] = poly_iou(pred_poly, gt_poly)
    
    # Find threshold that achieves target recall
    best_threshold = 0.0
    best_metrics = None
    
    for k in range(len(sorted_detections)):
        threshold = sorted_detections[k][1]
        current_detections = [det for det in sorted_detections if det[1] >= threshold]
        
        # Calculate metrics for current threshold
        matched_gts = set()
        match_ious = []
        tp = 0
        
        for i, (pred_poly, _) in enumerate(current_detections):
            best_match = -1
            best_iou = 0.0
            
            for j in range(len(gt_polygons)):
                if j in matched_gts:
                    continue
                if i < len(iou_matrix) and iou_matrix[i, j] >= iou_threshold and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_match = j
            
            if best_match >= 0:
                matched_gts.add(best_match)
                match_ious.append(best_iou)
                tp += 1
        
        fp = len(current_detections) - tp
        fn = len(gt_polygons) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(match_ious) if match_ious else 0.0
        
        # Check if we've reached target recall
        if recall >= target_recall or k == len(sorted_detections) - 1:
            best_metrics = {
                "precision_at_recall": precision,
                "actual_recall": recall,
                "f1_at_recall": f1,
                "mean_iou_at_recall": mean_iou,
                "threshold_for_recall": threshold
            }
            break
    
    return best_metrics or {
        "precision_at_recall": 0.0,
        "actual_recall": 0.0,
        "f1_at_recall": 0.0,
        "mean_iou_at_recall": 0.0,
        "threshold_for_recall": 0.0
    }


def calculate_safety_metrics(pred_polygons: List[List[List[int]]], 
                           gt_polygons: List[List[List[int]]]) -> Dict[str, float]:
    """
    Calculate safety metrics: word split rate, GT coverage rate, best IoU median.
    
    Returns:
        Dict containing safety metrics
    """
    if len(pred_polygons) == 0 and len(gt_polygons) == 0:
        return {
            "word_split_rate": 0.0,
            "gt_coverage_rate": 1.0,
            "best_iou_median": 1.0
        }
    
    if len(pred_polygons) == 0:
        return {
            "word_split_rate": 0.0,
            "gt_coverage_rate": 0.0,
            "best_iou_median": 0.0
        }
    
    if len(gt_polygons) == 0:
        return {
            "word_split_rate": 1.0,  # All predictions are word splits
            "gt_coverage_rate": 0.0,
            "best_iou_median": 0.0
        }
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_polygons), len(gt_polygons)))
    for i, pred_poly in enumerate(pred_polygons):
        for j, gt_poly in enumerate(gt_polygons):
            iou_matrix[i, j] = poly_iou(pred_poly, gt_poly)
    
    # Word split rate: ratio of predictions that have low IoU with all GTs
    split_threshold = 0.3  # Consider as word split if max IoU < 0.3
    word_splits = 0
    for i in range(len(pred_polygons)):
        max_iou_for_pred = np.max(iou_matrix[i, :]) if len(gt_polygons) > 0 else 0.0
        if max_iou_for_pred < split_threshold:
            word_splits += 1
    
    word_split_rate = word_splits / len(pred_polygons)
    
    # GT coverage rate: ratio of GTs that have reasonable IoU with at least one prediction
    coverage_threshold = 0.5
    covered_gts = 0
    for j in range(len(gt_polygons)):
        max_iou_for_gt = np.max(iou_matrix[:, j]) if len(pred_polygons) > 0 else 0.0
        if max_iou_for_gt >= coverage_threshold:
            covered_gts += 1
    
    gt_coverage_rate = covered_gts / len(gt_polygons)
    
    # Best IoU median: median of best IoU for each GT
    best_ious = []
    for j in range(len(gt_polygons)):
        best_iou = np.max(iou_matrix[:, j]) if len(pred_polygons) > 0 else 0.0
        best_ious.append(best_iou)
    
    best_iou_median = np.median(best_ious) if best_ious else 0.0
    
    return {
        "word_split_rate": word_split_rate,
        "gt_coverage_rate": gt_coverage_rate,
        "best_iou_median": best_iou_median
    }


def evaluate_params_with_score_sweep(det_path: str, 
                                    synth_image_paths: List[str], 
                                    real_image_paths: List[str],
                                    synth_gt_dict: Dict[str, List[List[List[int]]]], 
                                    real_gt_dict: Dict[str, List[List[List[int]]]],
                                    bin_thresh: float, unclip_ratio: float,
                                    real_weight: float = 0.8) -> Dict[str, float]:
    """
    Evaluate detector with given hyperparameters using score sweeping and dataset weighting.
    
    Args:
        det_path: Path to detector weights
        synth_image_paths: List of synthetic image paths
        real_image_paths: List of real image paths  
        synth_gt_dict: Synthetic ground truth dictionary
        real_gt_dict: Real ground truth dictionary
        bin_thresh: Binary threshold parameter
        unclip_ratio: Unclip ratio parameter
        real_weight: Weight for real dataset (synth weight = 1 - real_weight)
        
    Returns:
        Dict containing weighted aggregated metrics
    """
    # Load base config from file
    cfg = load_config("configs/post_process.yaml")
    
    # Override detector parameters with test values
    cfg["detector"]["bin_thresh"] = bin_thresh
    cfg["detector"]["unclip_ratio"] = unclip_ratio
    # Remove box_thresh dependency - we'll use score sweeping
    cfg["detector"]["box_thresh"] = 0.0  # Accept all detections, rely on scores
    
    # Load detector with new parameters
    device = pick_device()
    det = load_detector(det_path, cfg, device)
    
    def evaluate_dataset(image_paths, gt_dict, dataset_name):
        """Evaluate on a single dataset."""
        ap_scores = []
        recall_metrics = []
        safety_metrics = []
        skipped_count = 0
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            
            # Get ground truth
            gt_polygons = gt_dict.get(img_name, [])
            
            # Skip images without annotations
            if not gt_polygons:
                skipped_count += 1
                continue
            
            # Run detection with scores
            try:
                # Use the unified function that always returns both polygons and scores
                pred_polygons_list, scores_list = run_detection_only(det, img_path)
                if len(pred_polygons_list) > 0 and pred_polygons_list[0].size > 0:
                    pred_polygons = pred_polygons_list[0].tolist()
                    # scores_list[0] is always a valid numpy array from loc_to_polygons
                    scores = scores_list[0].tolist()
                else:
                    pred_polygons = []
                    scores = []
                    
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                pred_polygons = []
                scores = []
            
            # Create detections with scores
            detections_with_scores = [(poly, score) for poly, score in zip(pred_polygons, scores)]
            
            # Calculate AP@IoU=0.5 (with real objectness scores)
            ap_score, _ = calculate_ap_at_iou(detections_with_scores, gt_polygons, iou_threshold=0.5)
            ap_scores.append(ap_score)
            
            # Calculate metrics at fixed recall (95%) - now with real scores
            recall_metrics_item = calculate_metrics_at_fixed_recall(
                detections_with_scores, gt_polygons, target_recall=0.95, iou_threshold=0.5
            )
            recall_metrics.append(recall_metrics_item)
            
            # Calculate safety metrics
            safety_metrics_item = calculate_safety_metrics(pred_polygons, gt_polygons)
            safety_metrics.append(safety_metrics_item)
                
        # Aggregate metrics for this dataset
        if not ap_scores:
            return {
                "ap_iou_05": 0.0,
                "precision_at_95_recall": 0.0,
                "actual_recall": 0.0,
                "f1_at_95_recall": 0.0,
                "mean_iou_at_95_recall": 0.0,
                "word_split_rate": 1.0,
                "gt_coverage_rate": 0.0,
                "best_iou_median": 0.0
            }
        
        aggregated = {
            "ap_iou_05": np.mean(ap_scores),
            "precision_at_95_recall": np.mean([m["precision_at_recall"] for m in recall_metrics]),
            "actual_recall": np.mean([m["actual_recall"] for m in recall_metrics]),
            "f1_at_95_recall": np.mean([m["f1_at_recall"] for m in recall_metrics]),
            "mean_iou_at_95_recall": np.mean([m["mean_iou_at_recall"] for m in recall_metrics]),
            "word_split_rate": np.mean([m["word_split_rate"] for m in safety_metrics]),
            "gt_coverage_rate": np.mean([m["gt_coverage_rate"] for m in safety_metrics]),
            "best_iou_median": np.mean([m["best_iou_median"] for m in safety_metrics])
        }
        
        return aggregated
    
    # Evaluate on datasets based on real_weight
    if real_weight >= 1.0:
        # Skip synthetic dataset computation when real_weight is 1.0
        synth_metrics = {
            "ap_iou_05": 0.0,
            "precision_at_95_recall": 0.0,
            "actual_recall": 0.0,
            "f1_at_95_recall": 0.0,
            "mean_iou_at_95_recall": 0.0,
            "word_split_rate": 0.0,
            "gt_coverage_rate": 0.0,
            "best_iou_median": 0.0
        }
        real_metrics = evaluate_dataset(real_image_paths, real_gt_dict, "real")
    elif real_weight <= 0.0:
        # Skip real dataset computation when real_weight is 0.0
        real_metrics = {
            "ap_iou_05": 0.0,
            "precision_at_95_recall": 0.0,
            "actual_recall": 0.0,
            "f1_at_95_recall": 0.0,
            "mean_iou_at_95_recall": 0.0,
            "word_split_rate": 0.0,
            "gt_coverage_rate": 0.0,
            "best_iou_median": 0.0
        }
        synth_metrics = evaluate_dataset(synth_image_paths, synth_gt_dict, "synthetic")
    else:
        # Evaluate on both datasets
        synth_metrics = evaluate_dataset(synth_image_paths, synth_gt_dict, "synthetic")
        real_metrics = evaluate_dataset(real_image_paths, real_gt_dict, "real")
    
    # Calculate weighted metrics
    synth_weight = 1.0 - real_weight
    weighted_metrics = {}
    
    for key in synth_metrics.keys():
        weighted_metrics[key] = real_weight * real_metrics[key] + synth_weight * synth_metrics[key]
    
    # Add individual dataset metrics for reference
    weighted_metrics.update({
        f"synth_{key}": value for key, value in synth_metrics.items()
    })
    weighted_metrics.update({
        f"real_{key}": value for key, value in real_metrics.items()
    })
    
    return weighted_metrics


# Backward compatibility: keep old function name but redirect to new function
def evaluate_params(det_path: str, image_paths: List[str], 
                   gt_dict: Dict[str, List[List[List[int]]]], 
                   bin_thresh: float, unclip_ratio: float) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    This function assumes all images are synthetic.
    """
    return evaluate_params_with_score_sweep(
        det_path=det_path,
        synth_image_paths=image_paths,
        real_image_paths=[],
        synth_gt_dict=gt_dict,
        real_gt_dict={},
        bin_thresh=bin_thresh,
        unclip_ratio=unclip_ratio,
        real_weight=0.0  # All synthetic
    )


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


def run_phase(phase: int, det_path: str, 
              synth_image_paths: List[str], real_image_paths: List[str],
              synth_gt_dict: Dict[str, List[List[List[int]]]], 
              real_gt_dict: Dict[str, List[List[List[int]]]],
              best_params: Dict[str, float] = None,
              real_weight: float = 0.8) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Run a single optimization phase with new evaluation method.
    
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
    print(f"Dataset weighting: Real={real_weight:.1f}, Synthetic={1-real_weight:.1f}")
    
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
        # Evaluate current parameters with new method
        metrics = evaluate_params_with_score_sweep(
            det_path=det_path,
            synth_image_paths=synth_image_paths,
            real_image_paths=real_image_paths,
            synth_gt_dict=synth_gt_dict,
            real_gt_dict=real_gt_dict,
            bin_thresh=bin_thresh,
            unclip_ratio=unclip_ratio,
            real_weight=real_weight
        )
        
        # Store result
        result = {
            "unclip_ratio": float(unclip_ratio),
            "bin_thresh": float(bin_thresh),
            **metrics
        }
        results.append(result)
        
        # Update best result using new comparison criteria
        if best_result is None or compare_results_new(metrics, best_score) > 0:
            best_result = result.copy()
            best_score = metrics.copy()
        
        # Update progress bar description with new metrics (prioritizing AP now that we have real scores)
        progress_bar.set_postfix({
            'Best AP': f'{best_score["ap_iou_05"]:.4f}',
            'Best P@R95': f'{best_score["precision_at_95_recall"]:.4f}',
            'Best GTC': f'{best_score["gt_coverage_rate"]:.3f}'
        })
    
    elapsed_time = time.time() - start_time
    
    # Sort results by new performance criteria (AP first now that we have real scores)
    results.sort(key=lambda x: (x["ap_iou_05"], x["precision_at_95_recall"], x["gt_coverage_rate"], -x["word_split_rate"], x["best_iou_median"]), reverse=True)
    
    # Print results with new metrics
    print(f"\nPhase {phase} completed in {elapsed_time:.2f} seconds")
    print(f"Evaluated {len(results)} parameter combinations")
    print(f"\nTop 5 results:")
    print(f"{'Rank':<4} {'unclip_ratio':<12} {'bin_thresh':<10} {'AP@IoU0.5':<10} {'P@R95%':<8} {'GTC':<6} {'WSR':<6} {'BIoUM':<7}")
    print("-" * 85)
    
    for i, result in enumerate(results[:5]):
        print(f"{i+1:<4} {result['unclip_ratio']:<12.3f} {result['bin_thresh']:<10.3f} "
              f"{result['ap_iou_05']:<10.4f} {result['precision_at_95_recall']:<8.4f} "
              f"{result['gt_coverage_rate']:<6.3f} {result['word_split_rate']:<6.3f} "
              f"{result['best_iou_median']:<7.3f}")
    
    print(f"\nPhase {phase} best parameters:")
    print(f"unclip_ratio: {best_result['unclip_ratio']:.3f}")
    print(f"bin_thresh: {best_result['bin_thresh']:.3f}")
    print(f"AP@IoU=0.5: {best_result['ap_iou_05']:.4f} (now with real objectness scores)")
    print(f"Precision@Recall=95%: {best_result['precision_at_95_recall']:.4f}")
    print(f"GT Coverage Rate: {best_result['gt_coverage_rate']:.4f} (higher is better)")
    print(f"Word Split Rate: {best_result['word_split_rate']:.4f} (lower is better)")
    print(f"Best IoU Median: {best_result['best_iou_median']:.4f} (higher is better)")
    
    # Show dataset-specific results
    print(f"\nDataset-specific results:")
    print(f"Real dataset - AP: {best_result['real_ap_iou_05']:.4f}, P@R95%: {best_result['real_precision_at_95_recall']:.4f}")
    print(f"Synth dataset - AP: {best_result['synth_ap_iou_05']:.4f}, P@R95%: {best_result['synth_precision_at_95_recall']:.4f}")
    
    return best_result, results


def compare_results_new(result1: Dict[str, float], result2: Dict[str, float]) -> int:
    """
    Compare two results according to the new priority with real objectness scores:
    1. AP@IoU=0.5 (higher is better) - now meaningful with real scores
    2. Precision@Recall=95% (higher is better) - precision at high recall
    3. GT coverage rate (higher is better) - safety: ensures we don't miss ground truth
    4. Word split rate (lower is better) - safety valve against over-segmentation  
    5. Best IoU median (higher is better) - quality of detection
    
    Returns:
        1 if result1 > result2, -1 if result1 < result2, 0 if equal
    """
    # Compare AP@IoU=0.5 (higher is better) - now primary metric with real scores
    if result1["ap_iou_05"] > result2["ap_iou_05"]:
        return 1
    elif result1["ap_iou_05"] < result2["ap_iou_05"]:
        return -1
    
    # Compare Precision@Recall=95% (higher is better)
    if result1["precision_at_95_recall"] > result2["precision_at_95_recall"]:
        return 1
    elif result1["precision_at_95_recall"] < result2["precision_at_95_recall"]:
        return -1
    
    # Compare GT coverage rate (higher is better) - safety metric
    if result1["gt_coverage_rate"] > result2["gt_coverage_rate"]:
        return 1
    elif result1["gt_coverage_rate"] < result2["gt_coverage_rate"]:
        return -1
    
    # Compare word split rate (lower is better) - safety valve
    if result1["word_split_rate"] < result2["word_split_rate"]:
        return 1
    elif result1["word_split_rate"] > result2["word_split_rate"]:
        return -1
    
    # Compare best IoU median (higher is better) - quality metric
    if result1["best_iou_median"] > result2["best_iou_median"]:
        return 1
    elif result1["best_iou_median"] < result2["best_iou_median"]:
        return -1
    
    return 0


def main():
    args = parse_args()
    
    print("Starting detector hyperparameter optimization with new evaluation method")
    print(f"Detector model: {args.det_path}")
    print(f"Synthetic dataset: {args.input}")
    print(f"Real dataset: {args.input_real}")
    
    # Load test images from both datasets
    synth_image_paths = list_images(args.input)
    real_image_paths = list_images(args.input_real)
    print(f"Found {len(synth_image_paths)} synthetic test images")
    print(f"Found {len(real_image_paths)} real test images")
    
    # Load ground truth from both datasets
    synth_gt_dict = load_ground_truth(args.input)
    real_gt_dict = load_ground_truth(args.input_real)
    print(f"Loaded synthetic ground truth for {len(synth_gt_dict)} images")
    print(f"Loaded real ground truth for {len(real_gt_dict)} images")
    
    # Dataset weighting from command line argument
    real_weight = args.real_weight
    # Validate real_weight range
    if real_weight < 0.0 or real_weight > 1.0:
        raise ValueError(f"real_weight must be between 0.0 and 1.0, got {real_weight}")
    
    print(f"Using dataset weighting: Real={real_weight:.1f}, Synthetic={1-real_weight:.1f}")
    
    if real_weight >= 1.0:
        print("Note: Synthetic dataset evaluation will be skipped (real_weight=1.0)")
    elif real_weight <= 0.0:
        print("Note: Real dataset evaluation will be skipped (real_weight=0.0)")
    
    # Phase 1: Coarse search
    phase1_best, phase1_results = run_phase(
        1, args.det_path, synth_image_paths, real_image_paths,
        synth_gt_dict, real_gt_dict, real_weight=real_weight
    )
    
    # Phase 2: Fine search around phase 1 best
    phase2_best, phase2_results = run_phase(
        2, args.det_path, synth_image_paths, real_image_paths,
        synth_gt_dict, real_gt_dict, 
        best_params={"unclip_ratio": phase1_best["unclip_ratio"], 
                     "bin_thresh": phase1_best["bin_thresh"]},
        real_weight=real_weight
    )
    
    # Phase 3: Micro adjustment around phase 2 best (optional)
    if args.skip_phase3:
        print(f"\nSkipping Phase 3 (micro adjustment) as requested")
        phase3_best = phase2_best
        phase3_results = []
    else:
        phase3_best, phase3_results = run_phase(
            3, args.det_path, synth_image_paths, real_image_paths,
            synth_gt_dict, real_gt_dict,
            best_params={"unclip_ratio": phase2_best["unclip_ratio"], 
                         "bin_thresh": phase2_best["bin_thresh"]},
            real_weight=real_weight
        )
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL OPTIMIZATION RESULTS (New Evaluation Method)")
    print(f"{'='*80}")
    
    print(f"\nPhase 1 (Coarse) best:")
    print(f"  unclip_ratio: {phase1_best['unclip_ratio']:.3f}, bin_thresh: {phase1_best['bin_thresh']:.3f}")
    print(f"  AP@IoU=0.5: {phase1_best['ap_iou_05']:.4f}, P@R95%: {phase1_best['precision_at_95_recall']:.4f}")
    print(f"  WSR: {phase1_best['word_split_rate']:.4f}, GTC: {phase1_best['gt_coverage_rate']:.4f}, BIoUM: {phase1_best['best_iou_median']:.4f}")
    
    print(f"\nPhase 2 (Fine) best:")
    print(f"  unclip_ratio: {phase2_best['unclip_ratio']:.3f}, bin_thresh: {phase2_best['bin_thresh']:.3f}")
    print(f"  AP@IoU=0.5: {phase2_best['ap_iou_05']:.4f}, P@R95%: {phase2_best['precision_at_95_recall']:.4f}")
    print(f"  WSR: {phase2_best['word_split_rate']:.4f}, GTC: {phase2_best['gt_coverage_rate']:.4f}, BIoUM: {phase2_best['best_iou_median']:.4f}")
    
    if not args.skip_phase3:
        print(f"\nPhase 3 (Micro) best:")
        print(f"  unclip_ratio: {phase3_best['unclip_ratio']:.3f}, bin_thresh: {phase3_best['bin_thresh']:.3f}")
        print(f"  AP@IoU=0.5: {phase3_best['ap_iou_05']:.4f}, P@R95%: {phase3_best['precision_at_95_recall']:.4f}")
        print(f"  WSR: {phase3_best['word_split_rate']:.4f}, GTC: {phase3_best['gt_coverage_rate']:.4f}, BIoUM: {phase3_best['best_iou_median']:.4f}")
    else:
        print(f"\nPhase 3 (Micro) skipped")
    
    print(f"\nFINAL RECOMMENDED PARAMETERS:")
    print(f"unclip_ratio: {phase3_best['unclip_ratio']:.3f}")
    print(f"bin_thresh: {phase3_best['bin_thresh']:.3f}")
    
    # Performance comparison
    improvement_ap = phase3_best['ap_iou_05'] - phase1_best['ap_iou_05']
    improvement_p95 = phase3_best['precision_at_95_recall'] - phase1_best['precision_at_95_recall']
    improvement_wsr = phase1_best['word_split_rate'] - phase3_best['word_split_rate']  # Lower is better
    
    comparison_label = "PHASE 1 TO PHASE 3" if not args.skip_phase3 else "PHASE 1 TO PHASE 2"
    print(f"\nIMPROVEMENT FROM {comparison_label}:")
    print(f"AP@IoU=0.5: {improvement_ap:+.4f} ({'better' if improvement_ap > 0 else 'worse' if improvement_ap < 0 else 'same'})")
    print(f"Precision@R95%: {improvement_p95:+.4f} ({'better' if improvement_p95 > 0 else 'worse' if improvement_p95 < 0 else 'same'})")
    print(f"Word Split Rate: {improvement_wsr:+.4f} ({'better' if improvement_wsr > 0 else 'worse' if improvement_wsr < 0 else 'same'})")
    
    print(f"\nDETAILED DATASET-SPECIFIC RESULTS:")
    if real_weight > 0.0:
        print(f"Real Dataset (weight={real_weight:.1f}):")
        print(f"  AP@IoU=0.5: {phase3_best['real_ap_iou_05']:.4f}")
        print(f"  Precision@R95%: {phase3_best['real_precision_at_95_recall']:.4f}")
        print(f"  Word Split Rate: {phase3_best['real_word_split_rate']:.4f}")
        print(f"  GT Coverage Rate: {phase3_best['real_gt_coverage_rate']:.4f}")
    
    if real_weight < 1.0:
        print(f"Synthetic Dataset (weight={1-real_weight:.1f}):")
        print(f"  AP@IoU=0.5: {phase3_best['synth_ap_iou_05']:.4f}")
        print(f"  Precision@R95%: {phase3_best['synth_precision_at_95_recall']:.4f}")
        print(f"  Word Split Rate: {phase3_best['synth_word_split_rate']:.4f}")
        print(f"  GT Coverage Rate: {phase3_best['synth_gt_coverage_rate']:.4f}")

    print(f"\nIt is recommended to update unclip_ratio: {phase3_best['unclip_ratio']:.3f} and bin_thresh: {phase3_best['bin_thresh']:.3f} in configs/post_process.yaml")
    
    note_suffix = f" (Phase 3 skipped)" if args.skip_phase3 else ""
    weight_info = f" with real_weight={real_weight:.1f}"
    print(f"\nNote: This optimization uses score sweeping with real objectness scores (box_thresh independent) and weighted evaluation{weight_info}{note_suffix}.")
    print(f"WSR=Word Split Rate (lower is better), GTC=GT Coverage Rate (higher is better), BIoUM=Best IoU Median (higher is better)")
    
    if args.skip_phase3:
        print(f"\nTo get even better results, you can run without --skip_phase3 for phase 3 micro adjustment.")
    
    if real_weight >= 1.0:
        print(f"\nNote: Only real dataset was evaluated (real_weight=1.0). To include synthetic data, use --real_weight < 1.0")
    elif real_weight <= 0.0:
        print(f"\nNote: Only synthetic dataset was evaluated (real_weight=0.0). To include real data, use --real_weight > 0.0")


if __name__ == "__main__":
    main()
