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
from tqdm import tqdm

# ── docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection, recognition
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor

from shapely.geometry import Polygon

from .config import get_charset, get_model_config


# -------------------------
# Config / Device
# -------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # モデル設定も読み込み
    model_config = get_model_config()
    cfg["model"] = model_config
    
    # 文字セット設定も読み込み
    vocab = get_charset()
    cfg["charset"] = {"vocab": vocab}
    
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
    # モデル設定からアーキテクチャを取得
    det_arch = cfg["model"]["detector"]["arch"]
    det = detection.__dict__[det_arch](pretrained=False)
    
    ckpt = torch.load(det_pt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # references の保存形式と素の state_dict 両対応
    det.load_state_dict(state, strict=False)
    det.to(device).eval()

    pp_cfg = cfg["detector"]

    # postprocessorのハイパーパラメータ設定
    det.postprocessor = DBPostProcessor(
        bin_thresh=pp_cfg["bin_thresh"],
        box_thresh=pp_cfg["box_thresh"],
        assume_straight_pages=False
    )
    
    # コンストラクタで設定できない属性を後から設定
    det.postprocessor.unclip_ratio = pp_cfg["unclip_ratio"]
    # min_size は docTR の内部でハードコードされているため、ここでは設定不要

    return det

def load_recognizer(rec_pt: str, cfg: Dict[str, Any], device: torch.device):
    # モデル設定からアーキテクチャを取得
    rec_arch = cfg["model"]["recognizer"]["arch"]
    vocab = cfg["charset"]["vocab"]
    reco = recognition.__dict__[rec_arch](pretrained=False, vocab=vocab)
    
    ckpt = torch.load(rec_pt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    reco.load_state_dict(state, strict=False)
    reco.to(device).eval()
    # beam_width は必要なら cfg["recognizer"]["beam_width"] から取得して使う
    return reco

def build_predictor(det, reco):
    # doctr の ocr_predictor を使用
    return ocr_predictor(det_arch=det, reco_arch=reco, preserve_aspect_ratio=True, symmetric_pad=True, assume_straight_pages=False)


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
def run_inference_on_image(predictor, image_path: str, cfg: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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
                # word.geometry は正規化座標（0-1）
                poly_norm = word.geometry
                
                # tupleをnumpy arrayに変換
                if isinstance(poly_norm, tuple):
                    poly_norm = np.array(poly_norm)
                
                # 2点のbounding box形式を4点のpolygonに変換
                if poly_norm.shape[0] == 2:
                    # (x1,y1), (x2,y2) -> 4点のpolygon
                    x1, y1 = poly_norm[0]
                    x2, y2 = poly_norm[1]
                    # 左上 -> 右上 -> 右下 -> 左下 の順序
                    poly_norm = np.array([
                        [x1, y1],  # 左上
                        [x2, y1],  # 右上
                        [x2, y2],  # 右下
                        [x1, y2]   # 左下
                    ])
                
                # ピクセル座標に変換
                pts = [[int(x * w), int(y * h)] for (x, y) in poly_norm]
                conf = float(getattr(word, "confidence", 1.0))
                
                # 後段フィルタの適用
                text = word.value
                if cfg and "recognizer" in cfg:
                    rec_cfg = cfg["recognizer"]
                    
                    # min_conf フィルタ
                    min_conf = rec_cfg["min_conf"]
                    if conf < min_conf:
                        continue
                    
                    # min_len フィルタ
                    min_len = rec_cfg.get("min_len", 0)
                    if len(text) < min_len:
                        continue
                
                results.append({
                    "polygon": pts,
                    "text": text,
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
    
    # 画像サイズに基づいてフォントサイズを調整（プレゼンテーション用に大きく）
    img_height, img_width = img.shape[:2]
    base_font_size = max(1.2, min(img_width, img_height) / 800)  # 最小1.2、画像サイズに応じて調整
    
    for item in preds:
        pts = np.array(item["polygon"], dtype=np.int32)
        # ボックスの線を太くしてより見やすく
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=4)
        
        # ラベル描画は左上付近に
        x = min(p[0] for p in item["polygon"])
        y = min(p[1] for p in item["polygon"])
        
        # テキストサイズを大きくしてプレゼンテーション用に最適化
        font_scale = base_font_size
        thickness = max(2, int(base_font_size * 2))
        
        # テキストの背景を追加して読みやすくする
        text = f'{item["text"]}'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 背景矩形を描画
        bg_x1 = x - 2
        bg_y1 = max(0, y - text_height - baseline - 8)
        bg_x2 = x + text_width + 4
        bg_y2 = max(0, y - 2)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), 2)
        
        # テキストを描画
        cv2.putText(img, text, (x, max(text_height + 4, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness, cv2.LINE_AA)
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


# -------------------------
# Progress bar helpers
# -------------------------
def create_progress_bar(iterable, desc: str = "", total: int = None):
    """共通のプログレスバー作成関数"""
    return tqdm(iterable, desc=desc, total=total, unit="files", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

def run_inference_with_progress(predictor, image_paths: List[str], cfg: Dict[str, Any] = None, 
                               desc: str = "Processing images") -> Dict[str, List[Dict[str, Any]]]:
    """プログレスバー付きで複数画像の推論を実行"""
    results = {}
    for img_path in create_progress_bar(image_paths, desc=desc):
        try:
            preds = run_inference_on_image(predictor, img_path, cfg)
            results[os.path.basename(img_path)] = preds
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            results[os.path.basename(img_path)] = []
    return results
