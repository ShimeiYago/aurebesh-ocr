# scripts/ocr_common.py
from __future__ import annotations
import os
import glob
import json
from typing import Any, Dict, List, Tuple, Optional

import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

# ── docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models.detection import db_mobilenet_v3_large
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.recognition import crnn_mobilenet_v3_small

from shapely.geometry import Polygon

from .config import get_charset

from .constants import DETECTION_NORMALIZATION


# -------------------------
# Config / Device
# -------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
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
    det = db_mobilenet_v3_large(pretrained=False)
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
    vocab = cfg["charset"]["vocab"]
    reco = crnn_mobilenet_v3_small(pretrained=False, vocab=vocab)
    ckpt = torch.load(rec_pt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    reco.load_state_dict(state, strict=False)
    reco.to(device).eval()
    # beam_width は必要なら cfg["recognizer"]["beam_width"] から取得して使う
    return reco

def build_predictor(det, reco):
    # doctr の ocr_predictor を使用
    return ocr_predictor(det_arch=det, reco_arch=reco, assume_straight_pages=False)


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
# Detection-only preprocessing (letterbox + normalize)
# -------------------------
def _letterbox_resize(
    img_bgr: np.ndarray,
    out_size: int,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, int, int]:
    """
    OpenCVで学習時と同じ「正方キャンバスへのレターボックス」を再現。
    Return: resized_canvas_bgr, ratio, pad_left, pad_top
    """
    h0, w0 = img_bgr.shape[:2]
    if not preserve_aspect_ratio:
        resized = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return resized, out_size / w0, 0, 0

    r = min(out_size / w0, out_size / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((out_size, out_size, 3), pad_value, dtype=resized.dtype)
    if symmetric_pad:
        pad_x = (out_size - nw) // 2
        pad_y = (out_size - nh) // 2
    else:
        pad_x, pad_y = 0, 0

    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, r, pad_x, pad_y


def preprocess_for_detector(
    img_bgr: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    学習スクリプトと同一の前処理：
      1) レターボックス (preserve_aspect_ratio + symmetric_pad)
      2) BGR->RGB, 0-1
      3) Normalize(mean, std)
      4) CHW Tensor化
    戻り値: (1,3,S,S) tensor, meta辞書（元サイズ/スケール/パディングなど）
    """
    det_cfg = cfg["detector"]
    S = int(det_cfg.get("input_size", 1024))
    mean = np.array(DETECTION_NORMALIZATION["mean"], dtype=np.float32)
    std  = np.array(DETECTION_NORMALIZATION["std"], dtype=np.float32)
    keep = bool(det_cfg.get("preserve_aspect_ratio", True))
    sym  = bool(det_cfg.get("symmetric_pad", True))

    # letterbox
    canvas_bgr, ratio, pad_x, pad_y = _letterbox_resize(img_bgr, S, keep, sym, pad_value=0)

    # BGR->RGB, 0-1, Normalize
    img = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW

    tensor = torch.from_numpy(img).unsqueeze(0)  # 1,3,S,S
    meta = {
        "orig_wh": (img_bgr.shape[1], img_bgr.shape[0]),
        "proc_size": (S, S),
        "ratio": ratio,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return tensor, meta


# 検出出力（0-1座標）を元画像ピクセル座標に戻す関数
def _norm_poly_to_pixels(poly01: np.ndarray, meta: Dict[str, Any]) -> List[List[int]]:
    """
    poly01: 形状 (4,2) or (N,2) の 0-1 正規化座標（検出器の出力）
    meta: preprocess_for_detector が返した辞書
    """
    S_w, S_h = meta["proc_size"]
    r = meta["ratio"]; px = meta["pad_x"]; py = meta["pad_y"]
    W0, H0 = meta["orig_wh"]

    # 正方キャンバス上のピクセル座標へ
    xy = poly01.copy().astype(np.float32)
    xy[:, 0] *= S_w
    xy[:, 1] *= S_h

    # レターボックスのパディングを除去→元スケールへ
    xy[:, 0] = (xy[:, 0] - px) / max(r, 1e-6)
    xy[:, 1] = (xy[:, 1] - py) / max(r, 1e-6)

    # 画像外をクリップ
    xy[:, 0] = np.clip(xy[:, 0], 0, W0 - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, H0 - 1)

    return [[int(round(x)), int(round(y))] for x, y in xy]


# 画像1枚からポリゴンだけを得る検出ルーチン
@torch.no_grad()
def run_detection_on_image(det_model, image_path: str, cfg: Dict[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    """
    Return: [{"polygon": [[x,y],...], "score": float}, ...]
    ※ 認識は行わず、検出のみ
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    x, meta = preprocess_for_detector(img_bgr, cfg)
    x = x.to(device)

    # forward（doctrのDBNetは return_preds=True で正規化座標を返す）
    out = det_model(x, return_preds=True)
    loc_preds = out["preds"]  # List[Dict[str, np.ndarray]]
    results: List[Dict[str, Any]] = []

    if not loc_preds:
        return results

    # 通常は1画像につき1辞書（クラス名→(N,5,2) など）
    pred_dict = loc_preds[0]
    for boxes in pred_dict.values():
        # 形状の想定: (N, 5, 2) → 前4点がポリゴン、最後の点のx(or y)がscoreになる実装が多い
        # バージョン差に耐えるようにガード
        N = boxes.shape[0]
        for i in range(N):
            poly = boxes[i, :4, :] if boxes.ndim == 3 and boxes.shape[1] >= 4 else boxes[i]
            if poly.shape[0] == 2:
                # (x1,y1),(x2,y2) → 4点に展開
                (x1, y1), (x2, y2) = poly
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            score = 1.0
            if boxes.ndim == 3 and boxes.shape[1] >= 5:
                # doctrの実装では boxes[i,4,0] や boxes[i,4,1] にscoreを積むことがある
                score = float(np.clip(boxes[i, 4].max(), 0.0, 1.0))

            pts = _norm_poly_to_pixels(np.asarray(poly, dtype=np.float32), meta)
            results.append({"polygon": pts, "score": score})
    return results


# ポリゴンだけを描画する関数
def draw_polygons(img_path: str, dets: List[Dict[str, Any]]) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    for i, item in enumerate(dets):
        pts = np.array(item["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
        # indexだけ小さく表示（必要ならスコアも）
        x = min(p[0] for p in item["polygon"])
        y = min(p[1] for p in item["polygon"])
        cv2.putText(img, f'#{i}', (x, max(0, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return img


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
