# Aurebesh OCR

*A lightweight, reproducible pipeline that detects and recognises Aurebesh text in images using
docTR + PyTorch on Apple Silicon M4.*

---

## Features
| Stage | Model / Tooling | Highlights |
|-------|-----------------|------------|
| **Detection** | **DBNet + MobileNetV3-Large** | pretrained warm-start & fine-tune |
| **Recognition** | **CRNN + MobileNetV3-Small** | custom Aurebesh charset |
| **Data synth** | **SynthTIGER** | font-weighted sampling, optional solid / gradient BG |
| **Monitoring** | **TensorBoard** | scalars, PR-curves, sample overlays |

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
assets/fonts/fancy/    # decorative / graffiti
```

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
python scripts/generate_dataset.py --num_images 20000

# 2. fine-tune detector (60 epochs ≈ 18 h on M4)
python scripts/train_det.py

# 3. fine-tune recogniser (40 epochs ≈ 6 h)
python scripts/train_rec.py

# 4. Evaluate Performance
python scripts/evaluate.py

# 5. Run Inference
python scripts/inference.py --img_path examples/sample_real.jpg
```

---

## Author

[Shimei Yago](https://github.com/ShimeiYago)

---

## License
Code released under **MIT**.  
Fonts and background images remain the property of their respective authors and
**must not** be redistributed.

---
