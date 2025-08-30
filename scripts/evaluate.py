# scripts/evaluate.py
from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict, List, Tuple

from utils.inference_common import (
    load_config, pick_device, load_detector, load_recognizer, build_predictor,
    list_images, read_labels_json, run_inference_on_image, poly_iou
)

def parse_args():
    ap = argparse.ArgumentParser(description="OCR end-to-end evaluation (det+rec)")
    ap.add_argument("--input", required=True, help="dataset root containing images/ and labels.json")
    ap.add_argument("--det_path", default="outputs/detection/mobilenet_large.pt", help=".pt path for detector")
    ap.add_argument("--rec_path", default="outputs/recognition/mobilenet_small.pt", help=".pt path for recognizer")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--save_path", default="outputs/evaluate/results.json", help="File path to save results")
    return ap.parse_args()

# 簡易 1:1 マッチング（Hungarianを使わない貪欲法）
def match_detections(
    preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], iou_thresh: float
) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    """
    Return:
      matches: [(pred_idx, gt_idx, iou), ...]
      fp_idx:  未使用predのインデックス
      fn_idx:  未使用gtのインデックス
    """
    matches: List[Tuple[int,int,float]] = []
    used_p, used_g = set(), set()
    # GT ごとに最良 pred を割当（IoU 最大）
    for gi, g in enumerate(gts):
        best_pi, best_iou = -1, 0.0
        for pi, p in enumerate(preds):
            if pi in used_p:
                continue
            iou = poly_iou(p["polygon"], g["polygon"])
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0 and best_iou >= iou_thresh:
            matches.append((best_pi, gi, best_iou))
            used_p.add(best_pi)
            used_g.add(gi)
    fp = [i for i in range(len(preds)) if i not in used_p]
    fn = [j for j in range(len(gts)) if j not in used_g]
    return matches, fp, fn

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = pick_device()
    iou_thresh = cfg["detector"]["iou_thresh"]

    det = load_detector(args.det_path, cfg, device)
    reco = load_recognizer(args.rec_path, cfg, device)
    predictor = build_predictor(det, reco)

    labels = read_labels_json(args.input)
    img_dir = os.path.join(args.input, "images")

    # 集計器
    det_tp = det_fp = det_fn = 0
    iou_sum = 0.0
    iou_cnt = 0

    word_total = 0
    word_correct = 0

    e2e_tp = e2e_fp = e2e_fn = 0  # Strict: IoU>=th かつ 文字完全一致

    for fname, ann in labels.items():
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            # ラベルにあるが画像がない場合はスキップ
            continue

        preds = run_inference_on_image(predictor, img_path)

        # GT の整形（texts が無い or 長さ不一致に備えて保護）
        polys = ann.get("polygons", [])
        texts = ann.get("texts", [])
        gts: List[Dict[str, Any]] = []
        for i, poly in enumerate(polys):
            gt_text = texts[i] if i < len(texts) else ""
            gts.append({"polygon": poly, "text": gt_text})

        # 検出マッチング
        matches, fp_idx, fn_idx = match_detections(preds, gts, iou_thresh)

        det_tp += len(matches)
        det_fp += len(fp_idx)
        det_fn += len(fn_idx)

        for (pi, gi, iou) in matches:
            iou_sum += iou
            iou_cnt += 1

            pred_text = preds[pi]["text"]
            gt_text = gts[gi]["text"]
            word_total += 1
            if pred_text == gt_text:
                word_correct += 1
                e2e_tp += 1              # IoU 条件かつ文字一致
            else:
                e2e_fp += 1              # 予測はあるが文字が違う → E2E 的には FP
                e2e_fn += 1              # かつ GT は未検出扱い → E2E 的には FN

        # マッチしなかった分は E2E 観点では：
        #   - 予測のみ（fp_idx）→ E2E FP
        #   - GTのみ（fn_idx）→ E2E FN
        e2e_fp += len(fp_idx)
        e2e_fn += len(fn_idx)

    # Detection 指標
    det_p = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
    det_r = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0.0
    mean_iou = (iou_sum / iou_cnt) if iou_cnt > 0 else 0.0

    # Recognition 指標（マッチ成功ペアに対して）
    word_acc = (word_correct / word_total) if word_total > 0 else 0.0

    # End-to-End（Strict）
    e2e_p = e2e_tp / (e2e_tp + e2e_fp) if (e2e_tp + e2e_fp) > 0 else 0.0
    e2e_r = e2e_tp / (e2e_tp + e2e_fn) if (e2e_tp + e2e_fn) > 0 else 0.0
    e2e_f1 = 2 * e2e_p * e2e_r / (e2e_p + e2e_r) if (e2e_p + e2e_r) > 0 else 0.0

    # ── 結果出力
    print("\n=== Detection (polygon IoU: shapely) ===")
    print(f"TP: {det_tp}  FP: {det_fp}  FN: {det_fn}")
    print(f"Precision: {det_p:.4f}  Recall: {det_r:.4f}  F1: {det_f1:.4f}  mean IoU: {mean_iou:.4f}")

    print("\n=== Recognition (matched pairs, strict) ===")
    print(f"Word Accuracy: {word_acc:.4f}  (N={word_total})")

    print("\n=== End-to-End (strict) ===")
    print(f"Precision: {e2e_p:.4f}  Recall: {e2e_r:.4f}  F1: {e2e_f1:.4f}")

    # 結果をJSONで保存
    results = {
        "detection": {
            "tp": det_tp,
            "fp": det_fp,
            "fn": det_fn,
            "precision": det_p,
            "recall": det_r,
            "f1": det_f1,
            "mean_iou": mean_iou
        },
        "recognition": {
            "word_accuracy": word_acc,
            "total_words": word_total,
            "correct_words": word_correct
        },
        "end_to_end": {
            "tp": e2e_tp,
            "fp": e2e_fp,
            "fn": e2e_fn,
            "precision": e2e_p,
            "recall": e2e_r,
            "f1": e2e_f1
        }
    }

    # 保存ディレクトリを作成
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # JSONファイルに保存
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を {args.save_path} に保存しました。")

if __name__ == "__main__":
    main()
