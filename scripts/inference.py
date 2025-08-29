# scripts/inference.py
import os
import json
import argparse

from utils.inference_common import (
    load_config, pick_device, load_detector, load_recognizer, build_predictor,
    list_images, run_inference_on_image, draw_predictions, cv2
)

def parse_args():
    ap = argparse.ArgumentParser(description="OCR inference (det+rec)")
    ap.add_argument("--input_images", required=True, help="dir or a single image path")
    ap.add_argument("--det_path", default="outputs/detection/mobilenet_large.pt", help=".pt path for detector")
    ap.add_argument("--rec_path", default="outputs/recognition/mobilenet_small.pt", help=".pt path for recognizer")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--save_dir", default="outputs/inference")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = pick_device()

    det, post = load_detector(args.det_path, cfg, device)
    reco = load_recognizer(args.rec_path, cfg, device)
    predictor = build_predictor(det, reco, post)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    results = {}
    for img_path in list_images(args.input_images):
        preds = run_inference_on_image(predictor, img_path)
        results[os.path.basename(img_path)] = preds

        vis = draw_predictions(img_path, preds)
        out_path = os.path.join(
            args.save_dir, "images",
            os.path.splitext(os.path.basename(img_path))[0] + "_pred.jpg"
        )
        cv2.imwrite(out_path, vis)

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved results to: {args.save_dir}")

if __name__ == "__main__":
    main()
