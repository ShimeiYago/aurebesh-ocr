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
from PIL import Image

# ── docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection, recognition
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection.zoo import detection_predictor
from doctr.utils.geometry import detach_scores
from doctr import transforms as T
from torchvision.transforms.v2 import Normalize

from shapely.geometry import Polygon

from .config import get_charset, get_model_config
from .img_process import perspective_crop_polygon


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

def loc_to_polygons(loc_preds: List[Dict], origin_page_shapes: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    feature mapから最終的なpolygon座標とobjectness scoresを計算する関数
    
    Args:
        loc_preds: detection predictorからの出力（辞書形式）
        origin_page_shapes: 元画像のサイズ [(height, width), ...]
    
    Returns:
        Tuple of (List of polygon coordinates, List of objectness scores) for each page
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
    result_scores = []
    
    for page_polygons, page_scores, (orig_h, orig_w) in zip(loc_preds_processed, objectness_scores, origin_page_shapes):
        # ピクセル座標に変換
        pixel_polygons = []
        for poly in page_polygons:
            # poly は (4, 2) の形状: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            pixel_poly = np.array([[int(x * orig_w), int(y * orig_h)] for x, y in poly])
            pixel_polygons.append(pixel_poly)
        
        if pixel_polygons:
            result_polygons.append(np.array(pixel_polygons))
            result_scores.append(np.array(page_scores))
        else:
            result_polygons.append(np.empty((0, 4, 2), dtype=np.int32))
            result_scores.append(np.empty((0,), dtype=np.float32))
    
    return result_polygons, result_scores

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


def run_detection_only(det, image_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    画像パスからdetectorのみを実行してpolygon座標とobjectness scoresを返す関数
    build_predictorと同じ画像前処理（DocumentFile）を使用
    内部で2つの関数を呼び出す：extract_loc + loc_to_polygons
    
    Args:
        det: detection model (load_detector で作成したもの)
        image_path: 画像ファイルのパス
    
    Returns:
        Tuple of (List of polygon coordinates, List of objectness scores) for each page
        Each polygon element is numpy array of shape (N, 4, 2) for N detections
        Each score element is numpy array of shape (N,) for N detections
        Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] per detection
        座標はピクセル座標（整数）で返され、左上から時計回りの順序に正規化される
    """
    # build_predictorと同じ画像前処理を使用
    pages = DocumentFile.from_images(image_path)
    
    # 元の画像サイズを保存
    origin_page_shapes = [page.shape[:2] for page in pages]

    # 1. location predictionsを抽出
    loc_preds, _ = extract_loc(det, pages)

    # 2. location predictionsからpolygonとscoreを計算
    result_polygons, result_scores = loc_to_polygons(loc_preds, origin_page_shapes)
    
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
    
    return normalized_result_polygons, result_scores


def run_recognition(rec, image: Image.Image) -> str:
    """
    PIL画像からrecognizerのみを実行してテキストを返す関数
    train_recognizer.pyと完全に一致する前処理を適用
    
    Args:
        rec: recognition model (load_recognizer で作成したもの)
        image: PIL Image オブジェクト
    
    Returns:
        認識されたテキスト文字列
    """
    try:
        # モデルの設定から input_size と正規化パラメータを取得
        input_size = rec.cfg["input_shape"][1]  # Height from (C, H, W)
        mean, std = rec.cfg["mean"], rec.cfg["std"]
        
        # PIL ImageをNumPy配列に変換（HWC形式、uint8）
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # グレースケール画像の場合
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA画像の場合
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # DocTRのT.Resizeを直接使わずに、OpenCVで手動リサイズ
        # train_recognizer.pyと同じように (input_size, 4 * input_size) にリサイズ
        target_height = input_size
        target_width = 4 * input_size
        
        # アスペクト比を保持してリサイズ
        h, w = img_array.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > (target_width / target_height):
            # 幅が基準：幅をtarget_widthに合わせる
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # 高さが基準：高さをtarget_heightに合わせる
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # リサイズ
        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # パディングして目標サイズに合わせる
        pad_height = target_height - new_height
        pad_width = target_width - new_width
        
        # 中央に配置するためのパディング
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # パディング追加（白で埋める）
        padded_img = cv2.copyMakeBorder(
            resized_img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # テンソルに変換
        img_tensor = torch.from_numpy(padded_img).float() / 255.0  # [0,1]に正規化
        if len(img_tensor.shape) == 3:  # (H, W, C) -> (C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)  # バッチ次元を追加: (1, C, H, W)
        
        # 正規化を適用（train_recognizer.pyのL356相当）
        normalize_transform = Normalize(mean=mean, std=std)
        img_tensor = normalize_transform(img_tensor)
        
        # デバイスに移動
        device = next(rec.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # 推論実行
        with torch.no_grad():
            rec.eval()
            output = rec(img_tensor)
            
            # 出力の形式をチェック
            if isinstance(output, (list, tuple)):
                # リストまたはタプルの場合、最初の要素を使用
                logits = output[0] if len(output) > 0 else output
            elif isinstance(output, dict):
                # 辞書形式の場合、適切なキーからlogitsを取得
                if 'preds' in output:
                    # 'preds'キーがある場合、これは既にデコードされた予測結果
                    preds = output['preds']
                    if isinstance(preds, list) and len(preds) > 0:
                        # 最初の予測を取得
                        first_pred = preds[0]
                        if isinstance(first_pred, tuple) and len(first_pred) >= 1:
                            # (text, confidence) の形式
                            return first_pred[0]  # テキスト部分を返す
                        elif isinstance(first_pred, str):
                            return first_pred
                    return ""
                elif 'logits' in output:
                    logits = output['logits']
                elif 'output' in output:
                    logits = output['output']
                else:
                    # 最初の値を使用
                    logits = list(output.values())[0]
            else:
                logits = output
            
            # logitsがまだリストの場合の処理（念のため）
            if isinstance(logits, list):
                if len(logits) > 0 and isinstance(logits[0], tuple):
                    # 既にデコードされた結果の場合
                    return logits[0][0] if len(logits[0]) > 0 else ""
                elif len(logits) > 0:
                    logits = logits[0]  # 最初の要素を取得
                else:
                    return ""
            
            # CTC出力からテキストに変換
            # docTRのrecognition modelは通常logitsを返すので、適切にデコード
            if hasattr(rec, 'postprocessor'):
                # postprocessorがある場合はそれを使用
                decoded = rec.postprocessor(logits)
                if isinstance(decoded, list) and len(decoded) > 0:
                    return decoded[0] if isinstance(decoded[0], str) else ""
            else:
                # 手動でCTCデコード（簡易版）
                # logitsは通常 (batch_size, sequence_length, num_classes) の形状
                if len(logits.shape) == 3:
                    # 最も確率の高いクラスを選択
                    pred_indices = torch.argmax(logits, dim=2)
                    pred_indices = pred_indices.squeeze(0)  # バッチ次元を除去
                    
                    # CTCのblankトークン（通常は0番目）を除去し、連続する同じ文字を統合
                    if hasattr(rec, 'vocab'):
                        vocab = rec.vocab
                        decoded_chars = []
                        prev_idx = -1
                        
                        for idx in pred_indices:
                            idx = idx.item()
                            # blankトークン（0）をスキップし、連続する同じ文字を統合
                            if idx != 0 and idx != prev_idx:
                                if idx < len(vocab):
                                    decoded_chars.append(vocab[idx])
                            prev_idx = idx
                        
                        return ''.join(decoded_chars)
        
        return ""
        
    except Exception as e:
        # エラーの詳細を出力してデバッグを容易にする
        print(f"Error in run_recognition: {e}")
        import traceback
        traceback.print_exc()
        return ""


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
def run_inference_on_image(det, rec, image_path: str, cfg: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    detectorとrecognizerを別々に実行してOCR結果を返す関数
    
    Args:
        det: detection model (load_detector で作成したもの)
        rec: recognition model (load_recognizer で作成したもの)
        image_path: 画像ファイルのパス
        cfg: 設定辞書（フィルタリング用）
    
    Returns:
        [{polygon: [[x,y],...], text: str, confidence: float}, ...]
        polygon は画像ピクセル座標（整数）に変換して返す
    """
    # 1. detection実行
    detection_results, _ = run_detection_only(det, image_path)
    
    # 画像が複数ページある場合は最初のページのみ処理
    if len(detection_results) == 0:
        return []
    
    page_polygons = detection_results[0]  # 最初のページの検出結果
    if len(page_polygons) == 0:
        return []
    
    # 2. 元画像を読み込み
    original_image = Image.open(image_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    results: List[Dict[str, Any]] = []
    
    # 3. 各検出領域でrecognition実行
    for polygon in page_polygons:
        # polygon は (4, 2) の numpy array: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        polygon_list = polygon.tolist()  # List[List[int]] 形式に変換
        
        try:
            # 4. polygonを使って画像を切り取り
            cropped_image = perspective_crop_polygon(original_image, polygon_list)
            
            # 5. 切り取った画像でrecognition実行
            recognized_text = run_recognition(rec, cropped_image)
            
            # 6. confidence計算（認識モデルから直接取得できない場合は1.0をデフォルト）
            confidence = 1.0  # デフォルト値
            
            # 7. 後段フィルタの適用
            if cfg and "recognizer" in cfg:
                rec_cfg = cfg["recognizer"]
                
                # min_conf フィルタ
                min_conf = rec_cfg.get("min_conf", 0.0)
                if confidence < min_conf:
                    continue
                
                # min_len フィルタ
                min_len = rec_cfg.get("min_len", 0)
                if len(recognized_text) < min_len:
                    continue
            
            # 8. 結果に追加
            if recognized_text:  # 空文字列でない場合のみ追加
                results.append({
                    "polygon": polygon_list,
                    "text": recognized_text,
                    "confidence": confidence,
                })
                
        except Exception as e:
            # 個別の認識エラーは警告として出力し、処理を続行
            print(f"Warning: Failed to recognize text in polygon {polygon_list}: {e}")
            continue
    
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

def run_inference_with_progress(det, rec, image_paths: List[str], cfg: Dict[str, Any] = None, 
                               desc: str = "Processing images") -> Dict[str, List[Dict[str, Any]]]:
    """プログレスバー付きで複数画像の推論を実行"""
    results = {}
    for img_path in create_progress_bar(image_paths, desc=desc):
        try:
            preds = run_inference_on_image(det, rec, img_path, cfg)
            results[os.path.basename(img_path)] = preds
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            results[os.path.basename(img_path)] = []
    return results
