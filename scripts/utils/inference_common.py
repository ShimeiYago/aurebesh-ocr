# scripts/ocr_common.py
from __future__ import annotations
import os
import glob
import json
from typing import Any, Dict, List, Tuple

import yaml
import torch
import numpy as np
import cv2

# ── docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models.detection import db_mobilenet_v3_large
from doctr.models.detection.core import DetectionPostProcessor
from doctr.models.recognition import crnn_mobilenet_v3_small

from shapely.geometry import Polygon


# -------------------------
# Config / Device
# -------------------------
CHARSET_PATH = "configs/charset_aurebesh.yaml"

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 文字セット設定も読み込み
    with open(CHARSET_PATH, "r") as f:
        charset_cfg = yaml.safe_load(f)
    
    cfg["charset"] = charset_cfg
    return cfg

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Model loading
# -------------------------
def load_detector(det_pt: str, cfg: Dict[str, Any], device: torch.device):
    det = db_mobilenet_v3_large(pretrained=False)
    ckpt = torch.load(det_pt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # references の保存形式と素の state_dict 両対応
    det.load_state_dict(state, strict=False)
    det.to(device).eval()

    pp_cfg = cfg["detector"]
    post = DetectionPostProcessor(
        bin_thresh=pp_cfg["bin_thresh"],
        box_thresh=pp_cfg["box_thresh"],
        # unclip_ratio=pp_cfg["unclip_ratio"],
        # min_size=pp_cfg["min_size"],
        assume_straight_pages=False
    )
    return det, post

def load_recognizer(rec_pt: str, cfg: Dict[str, Any], device: torch.device):
    vocab = cfg["charset"]["vocab"]
    print(f"Loaded vocabulary for recognizer: {vocab}")
    reco = crnn_mobilenet_v3_small(pretrained=False, vocab=vocab)
    ckpt = torch.load(rec_pt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    reco.load_state_dict(state, strict=False)
    reco.to(device).eval()
    # beam_width は必要なら cfg["recognizer"]["beam_width"] から取得して使う
    return reco

def build_predictor(det, reco, post):
    # doctr の ocr_predictor はインスタンスを渡せる？
    # return ocr_predictor(det_arch=det, reco_arch=reco, pretrained=False, detector_postprocessor=post)
    return ocr_predictor(det_arch=det, reco_arch=reco, pretrained=False, assume_straight_pages=True)


# -------------------------
# I/O helpers
# -------------------------
def list_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        # サブフォルダ images/ にも対応
        img_dir = os.path.join(input_path, "images")
        if os.path.isdir(img_dir):
            for e in exts:
                files.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(set(files))
    else:
        return [input_path]

def read_labels_json(dataset_root: str) -> Dict[str, Any]:
    fp = os.path.join(dataset_root, "labels.json")
    with open(fp, "r") as f:
        return json.load(f)


# -------------------------
# Inference core
# -------------------------
def run_inference_on_image(predictor, image_path: str) -> List[Dict[str, Any]]:
    """
    Return: [{polygon: [[x,y],...], text: str, confidence: float}, ...]
    polygon は画像ピクセル座標（整数）に変換して返す
    """
    doc = DocumentFile.from_images(image_path)
    out = predictor(doc)  # list-like; 1ページ想定
    # 読み込み済のサイズ取得（DocumentFile は内部で読むので別途 cv2 でもOK）
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    results: List[Dict[str, Any]] = []
    # doctr の階層: pages -> blocks -> lines -> words
    page = out.pages[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                # word.geometry は 0..1 の正規化 polygon（N×2）
                poly_norm = word.geometry
                pts = [[int(x * w), int(y * h)] for (x, y) in poly_norm]
                conf = float(getattr(word, "confidence", 1.0))
                results.append({
                    "polygon": pts,
                    "text": word.value,
                    "confidence": conf,
                })
    return results


# -------------------------
# Visualization
# -------------------------
def draw_predictions(img_path: str, preds: List[Dict[str, Any]]) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    for item in preds:
        pts = np.array(item["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
        # ラベル描画は左上付近に
        x = min(p[0] for p in item["polygon"])
        y = min(p[1] for p in item["polygon"])
        cv2.putText(img, f'{item["text"]}', (x, max(0, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    return img


# -------------------------
# Polygon IoU (厳密: shapely)
# -------------------------
def _to_valid_polygon(coords: List[List[int]]):
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # 自己交差などを修復
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None

def poly_iou(poly_a: List[List[int]], poly_b: List[List[int]]) -> float:
    pa = _to_valid_polygon(poly_a)
    pb = _to_valid_polygon(poly_b)
    if pa is None or pb is None:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return float(inter / union) if union > 0 else 0.0
