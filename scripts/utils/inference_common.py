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
from doctr.models.detection.zoo import detection_predictor
from doctr.utils.geometry import detach_scores

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
    return ocr_predictor(
        det_arch=det,
        reco_arch=reco,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        assume_straight_pages=False,
        disable_page_orientation=True,  # pytorchのOCRPredictor内でassume_horizontal=Trueにするために必要
    )

def extract_loc(det, pages: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    detectorからpure feature mapを抽出する関数
    
    Args:
        det: detection model (load_detector で作成したもの)
        pages: list of images as numpy arrays (H, W, 3)
    
    Returns:
        Tuple of (loc_preds, out_maps)
        - loc_preds: 正規化座標でのdetection結果
        - out_maps: raw feature maps from detector
    """
    # Dimension check (OCRPredictorと同じ)
    if any(page.ndim != 3 for page in pages):
        raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")
    
    # DetectionPredictorを直接使って推論
    # detection_predictorラッパーを作成（OCRPredictorと同じ設定）
    det_predictor = detection_predictor(
        det,
        batch_size=2,  # デフォルト値
        assume_straight_pages=False,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )
    
    # Detection実行 (OCRPredictorのL82相当)
    loc_preds, out_maps = det_predictor(pages, return_maps=True)
    
    return loc_preds, out_maps

def loc_to_polygons(loc_preds: List[Dict], origin_page_shapes: List[Tuple[int, int]]) -> List[np.ndarray]:
    """
    feature mapから最終的なpolygon座標を計算する関数
    
    Args:
        loc_preds: detection predictorからの出力（辞書形式）
        origin_page_shapes: 元画像のサイズ [(height, width), ...]
    
    Returns:
        List of polygon coordinates for each page (pixel coordinates)
    """
    # OCRPredictorのL105-109相当: 辞書形式から座標を抽出
    assert all(len(loc_pred) == 1 for loc_pred in loc_preds), (
        "Detection Model should output only one class"
    )
    
    loc_preds_processed = [list(loc_pred.values())[0] for loc_pred in loc_preds]
    
    # objectness scoresを分離 (OCRPredictorのL110相当)
    loc_preds_processed, objectness_scores = detach_scores(loc_preds_processed)
    
    # 正規化座標（0-1）をピクセル座標に変換
    result_polygons = []
    for page_polygons, (orig_h, orig_w) in zip(loc_preds_processed, origin_page_shapes):
        # ピクセル座標に変換
        pixel_polygons = []
        for poly in page_polygons:
            # poly は (4, 2) の形状: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            pixel_poly = np.array([[int(x * orig_w), int(y * orig_h)] for x, y in poly])
            pixel_polygons.append(pixel_poly)
        
        if pixel_polygons:
            result_polygons.append(np.array(pixel_polygons))
        else:
            result_polygons.append(np.empty((0, 4, 2), dtype=np.int32))
    
    return result_polygons

def normalize_polygon_order(polygon: np.ndarray) -> np.ndarray:
    """
    Polygon座標を左上から時計回りの順序に正規化する
    
    Args:
        polygon: (4, 2) shaped array of polygon coordinates
    
    Returns:
        Normalized polygon in order: top-left, top-right, bottom-right, bottom-left
    """
    # 重心を計算
    center = np.mean(polygon, axis=0)
    
    # 各点から重心への角度を計算
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    
    # 角度順でソート（左上から時計回り）
    sorted_indices = np.argsort(angles)
    
    # 最初の点が最も左上に近い点になるように調整
    # 各点のy座標 + x座標の和が最小の点を開始点とする
    scores = polygon[:, 0] + polygon[:, 1]  # top-left bias
    start_idx = np.argmin(scores)
    
    # start_idxを含む並び順に調整
    if start_idx in sorted_indices:
        start_pos = np.where(sorted_indices == start_idx)[0][0]
        sorted_indices = np.roll(sorted_indices, -start_pos)
    
    return polygon[sorted_indices]


def run_detection_only(det, image_path: str) -> List[np.ndarray]:
    """
    画像パスからdetectorのみを実行してpolygon座標を返す関数
    build_predictorと同じ画像前処理（DocumentFile）を使用
    内部で2つの関数を呼び出す：extract_loc + loc_to_polygons
    
    Args:
        det: detection model (load_detector で作成したもの)
        image_path: 画像ファイルのパス
    
    Returns:
        List of polygon coordinates for each page
        Each element is numpy array of shape (N, 4, 2) for N detections
        Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] per detection
        座標はピクセル座標（整数）で返され、左上から時計回りの順序に正規化される
    """
    # build_predictorと同じ画像前処理を使用
    pages = DocumentFile.from_images(image_path)
    
    # 元の画像サイズを保存
    origin_page_shapes = [page.shape[:2] for page in pages]

    # 1. location predictionsを抽出
    loc_preds, _ = extract_loc(det, pages)

    # 2. location predictionsからpolygonを計算
    result_polygons = loc_to_polygons(loc_preds, origin_page_shapes)
    
    # 3. 各polygonの座標順序を正規化
    normalized_result_polygons = []
    for page_polygons in result_polygons:
        if len(page_polygons) > 0:
            normalized_polygons = []
            for polygon in page_polygons:
                normalized_poly = normalize_polygon_order(polygon)
                normalized_polygons.append(normalized_poly)
            normalized_result_polygons.append(np.array(normalized_polygons))
        else:
            normalized_result_polygons.append(page_polygons)
    
    return normalized_result_polygons


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
