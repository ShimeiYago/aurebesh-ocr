# scripts/inference.py
import os
import json
import argparse

from utils.inference_common import (
    draw_polygons, load_config, pick_device, load_detector, load_recognizer, build_predictor,
    list_images, run_detection_on_image, run_full_ocr_on_image, draw_predictions, cv2, create_progress_bar
)

def parse_args():
    ap = argparse.ArgumentParser(description="OCR inference (det+rec)")
    ap.add_argument("--input_images", required=True, help="dir or a single image path")
    ap.add_argument("--det_path", default="outputs/detection/mobilenet_large.pt", help=".pt path for detector")
    ap.add_argument("--rec_path", default="outputs/recognition/mobilenet_small.pt", help=".pt path for recognizer")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--save_dir", default="outputs/inference")
    ap.add_argument("--input_size", type=int, default=32, help="Recognition input height (width = 4*height)")
    ap.add_argument("--detection_only", action="store_true", help="Run detection only (skip recognition)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = pick_device()

    det = load_detector(args.det_path, cfg, device)
    reco = load_recognizer(args.rec_path, cfg, device) if not args.detection_only else None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    image_paths = list_images(args.input_images)
    results = {}
    
    if args.detection_only:
        # Detection only mode
        for img_path in create_progress_bar(image_paths, desc="Detecting polygons"):
            # Run custom detection function (same preprocessing as training)
            dets = run_detection_on_image(det, img_path, cfg, device)
            results[os.path.basename(img_path)] = dets

            # Draw polygons only
            vis = draw_polygons(img_path, dets)
            out_path = os.path.join(
                args.save_dir, "images",
                os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
            )
            cv2.imwrite(out_path, vis)
    else:
        # Full OCR mode (detection + recognition)
        for img_path in create_progress_bar(image_paths, desc="Running OCR"):
            # Integrated detection + recognition flow
            ocr_results = run_full_ocr_on_image(det, reco, img_path, cfg, device, args.input_size)
            results[os.path.basename(img_path)] = ocr_results

            # Draw results with text annotations
            vis = draw_predictions(img_path, ocr_results)
            out_path = os.path.join(
                args.save_dir, "images",
                os.path.splitext(os.path.basename(img_path))[0] + "_ocr.jpg"
            )
            cv2.imwrite(out_path, vis)

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved results to: {args.save_dir}")
    if not args.detection_only:
        print(f"[INFO] Processed {len(image_paths)} images with full OCR")
        total_detections = sum(len(r) for r in results.values())
        print(f"[INFO] Total text regions detected and recognized: {total_detections}")
    else:
        print(f"[INFO] Processed {len(image_paths)} images with detection only")
        total_detections = sum(len(r) for r in results.values())
        print(f"[INFO] Total text regions detected: {total_detections}")

if __name__ == "__main__":
    main()
