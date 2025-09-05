# Aurebesh OCR

*A lightweight, reproducible pipeline that detects and recognises Aurebesh text in images using
docTR + PyTorch on Apple Silicon M4.*

---

## Features
| Stage | Model / Tooling | Highlights |
|-------|-----------------|------------|
| **Detection** | **DBNet + MobileNetV3-Large** | pretrained warm-start & fine-tune |
| **Recognition** | **CRNN + MobileNetV3-Small** | custom Aurebesh charset |

---

## Requirements

- Python 3.12.11  
- Apple Silicon Mac (M1–M4) with macOS
- ≈ 8 GB free GPU memory recommended

---

## Installation

1. **Set up Python environment**
   ```bash
   python --version  # Confirm Python version is 3.12.11
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Preparation

### 1. Place Aurebesh font files

Copy your `.otf / .ttf` fonts into the style buckets:

```
assets/fonts/core/     # canonical fonts
assets/fonts/variant/  # bold / italic / condensed
```

**Recommended fonts**

| Category | Font (Style)                     | License                                     | Download                                                              |
| -------- | -------------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| Core     | **Aurebesh AF – Canon**          | Public Domain                               | [FontSpace](https://www.fontspace.com/aurebesh-af-font-f49637)        |
|          | **FT Aurebesh – Regular**        | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **FT Aurebesh – UltraLight**     | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **Aurek-Besh – Regular**         | Freeware                                    | [FontSpace](https://www.fontspace.com/aurek-besh-font-f9639)          |
| Variant  | **FT Aurebesh – Black**          | SIL OFL 1.1                                 | [DeeFont](https://www.deefont.com/ft-aurebesh-font-family/)           |
|          | **Aurebesh Font – Italic**       | Freeware, commercial use requires donation  | [FontSpace](https://www.fontspace.com/aurebesh-font-f17959)           |
|          | **Aurek-Besh – Narrow**          | Freeware                                    | [FontSpace](https://www.fontspace.com/aurek-besh-font-f9639)          |


### 2. Place background images

Put any royalty-free photos (≥ 512 px) in:

```
assets/backgrounds/
```

### 3. Place test images

Place real photos for test and update `annotations.json`.

```
data/real/images/           # Some real photos
data/real/annotations.json  # COCO polygons + transcripts per image
```

---

## Usage

```bash
# 1. synthetic dataset (20 k images)
python scripts/generate_dataset.py --num_images 30000 --max_workers 6

# 2. Train detector (30 epochs)
python scripts/train_detector.py db_mobilenet_v3_large \
  --name mobilenet_large \
  --train_path data/synth/train \
  --val_path data/synth/val \
  --epochs 30 \
  --batch_size 4 \
  --input_size 1024 \
  --lr 0.0003 \
  --optim adamw \
  --wd 0.0001 \
  --sched cosine \
  --pretrained \
  --rotation \
  --eval-straight \
  --early-stop \
  --output_dir outputs/detector

# 3. Evaluate detector
python scripts/train_detector.py db_mobilenet_v3_large \
  --train_path data/synth/train \
  --val_path data/synth/test \
  --batch_size 4 \
  --input_size 1024 \
  --rotation \
  --test-only \
  --resume outputs/detection/mobilenet_large.pt

# 4. Train recognizer (50 epochs)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_recognizer.py crnn_mobilenet_v3_small \
  --name mobilenet_small \
  --train_path data/synth/train/cropped \
  --val_path data/synth/val/cropped \
  --epochs 50 \
  --batch_size 64 \
  --input_size 32 \
  --lr 0.003 \
  --optim adamw \
  --wd 0.0001 \
  --sched cosine \
  --vocab aurebesh \
  --early-stop \
  --early-stop-epochs 10 \
  --early-stop-delta 0.002 \
  --min-chars 1 \
  --max-chars 15 \
  --output_dir outputs/recognizer

# 5. Evaluate recognizer
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_recognizer.py crnn_mobilenet_v3_small \
  --train_path data/synth/train/cropped \
  --val_path data/synth/test/cropped \
  --batch_size 64 \
  --input_size 32 \
  --vocab aurebesh \
  --test-only \
  --resume outputs/recognition/mobilenet_small.pt

# 6. Evaluate E2E Performance
python scripts/evaluate.py \
  --input data/synth/test \
  --det_path outputs/detector/mobilenet_large.pt \
  --rec_path outputs/recognizer/mobilenet_small.pt \
  --config configs/inference.yaml \
  --save_path outputs/evaluate/results.json

# 7. Run Inference
python scripts/inference.py \
  --input_images data/real/images \
  --det_path outputs/detector/mobilenet_large.pt \
  --rec_path outputs/recognizer/mobilenet_small.pt \
  --config configs/inference.yaml \
  --save_dir outputs/inference
```

---

## Monitoring

To monitor training progress, launch TensorBoard with your log directory:

```bash
tensorboard --logdir <path_to_log_dir>
```

Replace `<path_to_log_dir>` with the actual path to your output logs (e.g., `outputs/log/your_run/tensorboard_dir`).

## Author

[Shimei Yago](https://github.com/ShimeiYago)

---

## License
Code released under **MIT**.  
Fonts and background images remain the property of their respective authors and
**must not** be redistributed.

---
