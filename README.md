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
| Fancy    | **Naboo AF Aurebesh**            | Public Domain                               | [FontSpace](https://www.fontspace.com/naboo-af-aurebesh-font-f118825) |
|          | **Droidobesh Depot**             | Public Domain                               | [FontSpace](https://www.fontspace.com/droidobesh-depot-font-f55049)   |
|          | **Aurebesh Racer AF – Regular**  | CC0 Public Domain                           | [CufonFonts](https://www.cufonfonts.com/font/aurebesh-racer-af)       |


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
python scripts/generate_dataset.py --num_images 30000

# 2. fine-tune detector (60 epochs)
python scripts/train_det.py

# 3. fine-tune recogniser (60 epochs)
python scripts/train_rec.py

# 4.1. Evaluate Performance for test data
python scripts/evaluate.py --images data/synth/test

# 4.2. Evaluate Performance for real data
python scripts/evaluate.py --images data/real

# 5. Run Inference
python scripts/inference.py --output_dir outputs/inference --img_path <real_img_path>
```

---

## Configuration

You can customize the behavior of the OCR pipeline by editing the configuration files in the `configs/` directory:

### `charset_aurebesh.yaml`
- **Purpose**: Defines the character set for recognition
- **Default**: Includes digits (0-9) and uppercase letters (A-Z)
- **Usage**: Modify to add/remove characters the recognizer should learn

### `dataset.yaml`
- **Purpose**: Controls synthetic dataset generation parameters
- **Key settings**:
  - `style.font`: Font sampling probabilities (core: 60%, variant: 30%, fancy: 10%)
  - `style.text`: Text generation parameters (word count, length ranges)
  - `style.color`: Text and background color modes
  - `effects`: Visual effects like shadow, border, rotation
  - `augmentation`: Data augmentation settings (perspective, noise, blur, JPEG compression)

### `train_det.yaml`
- **Purpose**: Detection model training configuration
- **Key settings**:
  - `epochs`: 60 (default)
  - `batch_size`: 4 with gradient accumulation
  - `optimizer`: AdamW with learning rate 1e-4
  - `scheduler`: CosineAnnealingLR with 5-epoch warmup
  - `amp`: Automatic Mixed Precision enabled

### `train_rec.yaml`
- **Purpose**: Recognition model training configuration
- **Key settings**:
  - `epochs`: 40 (default)
  - `batch_size`: 64
  - `optimizer`: AdamW with learning rate 2e-4
  - `scheduler`: OneCycleLR with 30% warmup
  - `metrics`: Character Error Rate (CER)
  - `charset`: Links to `charset_aurebesh.yaml`

To modify training behavior, edit the relevant YAML file before running the training scripts.

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
