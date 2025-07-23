#!/usr/bin/env python3
"""
Generate synthetic Aurebesh dataset using SynthTIGER.
Creates images with text detection annotations and LMDB for recognition.
"""

import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import albumentations as A
from tqdm import tqdm
import lmdb
import pickle

from utils import setup_logger, ensure_dir, load_config, get_charset


class AurebeshDatasetGenerator:
    def __init__(
        self,
        output_dir: Path,
        num_images: int = 20000,
        resolution: int = 1024,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        config_path: Optional[Path] = None
    ):
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.resolution = resolution
        self.split_ratio = split_ratio
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "dataset.yaml"
        self.config = load_config(config_path)
        
        # Setup paths
        self.font_dir = Path(__file__).parent.parent / "assets" / "fonts"
        self.bg_dir = Path(__file__).parent.parent / "assets" / "backgrounds"
        
        # Load resources
        self.fonts = self._load_fonts()
        self.backgrounds = list(self.bg_dir.glob("*.jpg")) + list(self.bg_dir.glob("*.png"))
        self.charset = get_charset()
        
        # Setup logger
        self.logger = setup_logger("generate_dataset", self.output_dir / "logs")
        
        # Setup augmentations
        self.augmentations = self._setup_augmentations()
        
    def _load_fonts(self) -> Dict[str, List[Path]]:
        """Load fonts from assets/fonts directory."""
        fonts = {
            'core': list((self.font_dir / 'core').glob('*.ttf')) + 
                   list((self.font_dir / 'core').glob('*.otf')),
            'variant': list((self.font_dir / 'variant').glob('*.ttf')) + 
                      list((self.font_dir / 'variant').glob('*.otf')),
            'fancy': list((self.font_dir / 'fancy').glob('*.ttf')) + 
                    list((self.font_dir / 'fancy').glob('*.otf'))
        }
        return fonts
    
    def _setup_augmentations(self) -> A.Compose:
        """Setup albumentations pipeline."""
        aug_config = self.config['augmentation']
        
        transforms = []
        
        # Perspective transform
        if aug_config['perspective']['prob'] > 0:
            deg_range = aug_config['perspective']['degree_range']
            transforms.append(
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=aug_config['perspective']['prob']
                )
            )
        
        # Noise
        if aug_config['noise']['prob'] > 0:
            transforms.append(
                A.GaussNoise(
                    var_limit=aug_config['noise']['var_range'],
                    p=aug_config['noise']['prob']
                )
            )
        
        # Blur
        if aug_config['blur']['prob'] > 0:
            kernel_range = aug_config['blur']['kernel_range']
            transforms.append(
                A.Blur(
                    blur_limit=kernel_range,
                    p=aug_config['blur']['prob']
                )
            )
        
        # JPEG compression
        if aug_config['jpeg']['prob'] > 0:
            quality_range = aug_config['jpeg']['quality_range']
            transforms.append(
                A.ImageCompression(
                    quality_lower=quality_range[0],
                    quality_upper=quality_range[1],
                    p=aug_config['jpeg']['prob']
                )
            )
        
        return A.Compose(transforms)
    
    def _sample_font(self) -> Path:
        """Sample font based on configured probabilities."""
        probs = self.config['style']['font']
        category = random.choices(
            ['core', 'variant', 'fancy'],
            weights=[probs['core_prob'], probs['variant_prob'], probs['fancy_prob']]
        )[0]
        return random.choice(self.fonts[category])
    
    def _generate_text(self) -> str:
        """Generate random Aurebesh text."""
        text_config = self.config['style']['text']
        num_words = random.randint(text_config['min_words'], text_config['max_words'])
        
        words = []
        for _ in range(num_words):
            word_len = random.randint(
                text_config['min_word_length'], 
                text_config['max_word_length']
            )
            # Use only letters from charset (no spaces or numbers for individual words)
            available_chars = [c for c in self.charset if c.isalpha()]
            word = ''.join(random.choices(available_chars, k=word_len))
            words.append(word)
        
        return ' '.join(words)
    
    def _get_background(self, size: Tuple[int, int]) -> Image.Image:
        """Get background image or generate synthetic one."""
        if random.random() < self.config['style']['color']['synthetic_bg_prob']:
            # Generate synthetic background
            bg = Image.new('RGB', size)
            draw = ImageDraw.Draw(bg)
            
            if random.random() < 0.5:
                # Solid color
                color = tuple(random.randint(0, 255) for _ in range(3))
                draw.rectangle([0, 0, size[0], size[1]], fill=color)
            else:
                # Gradient
                for y in range(size[1]):
                    color = tuple(
                        int(128 + 127 * np.sin(y / size[1] * np.pi + random.random() * 2 * np.pi))
                        for _ in range(3)
                    )
                    draw.line([(0, y), (size[0], y)], fill=color)
            
            return bg
        else:
            # Use real background
            bg_path = random.choice(self.backgrounds)
            bg = Image.open(bg_path).convert('RGB')
            bg = bg.resize(size, Image.Resampling.LANCZOS)
            return bg
    
    def _render_text_on_image(
        self, 
        text: str, 
        font_path: Path, 
        image_size: Tuple[int, int]
    ) -> Tuple[Image.Image, List[Dict]]:
        """Render multiple text blocks on image and return image with annotations."""
        # Create background
        bg = self._get_background(image_size)
        # Convert to RGBA for rotation support
        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')
        
        # Create drawing context
        draw = ImageDraw.Draw(bg)
        
        # Generate multiple text blocks (1-5 per image)
        num_text_blocks = random.randint(1, 5)
        annotations = []
        occupied_regions = []  # To avoid overlapping text
        
        for block_idx in range(num_text_blocks):
            # Generate text for this block (or use provided text for first block)
            if block_idx == 0:
                block_text = text
            else:
                block_text = self._generate_text()
            
            # Sample font for this block (can vary per block)
            if block_idx > 0:
                font_path = self._sample_font()
            
            # Setup font with varying sizes
            font_size = random.randint(30, 80)
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except:
                self.logger.warning(f"Failed to load font {font_path}, using default")
                font = ImageFont.load_default()
            
            # Text color for this block
            text_color = tuple(random.randint(0, 255) for _ in range(3))
            
            # Calculate text dimensions
            bbox = draw.textbbox((0, 0), block_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Try to find non-overlapping position
            max_attempts = 50
            placed = False
            
            for attempt in range(max_attempts):
                max_x = max(10, image_size[0] - text_width - 10)
                max_y = max(10, image_size[1] - text_height - 10)
                
                if max_x <= 10 or max_y <= 10:
                    break  # Text too large for image
                
                x = random.randint(10, max_x)
                y = random.randint(10, max_y)
                
                # Check overlap with existing text blocks
                new_region = [x - 5, y - 5, x + text_width + 5, y + text_height + 5]  # Add padding
                overlap = False
                
                for region in occupied_regions:
                    if (new_region[0] < region[2] and new_region[2] > region[0] and
                        new_region[1] < region[3] and new_region[3] > region[1]):
                        overlap = True
                        break
                
                if not overlap or attempt == max_attempts - 1:
                    # Found non-overlapping position or last attempt
                    occupied_regions.append(new_region)
                    placed = True
                    
                    # Draw text with optional effects
                    effects = self.config['style']['effects']
                    
                    # Check if we should apply rotation
                    if 'rotation_prob' in effects and random.random() < effects['rotation_prob']:
                        # Apply rotation
                        rotation_range = effects.get('rotation_range', [-15, 15])
                        angle = random.uniform(rotation_range[0], rotation_range[1])
                        
                        # Create a temporary image for the text
                        text_img = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))
                        text_draw = ImageDraw.Draw(text_img)
                        
                        # Draw text with effects on temporary image
                        # Shadow
                        if random.random() < effects['shadow_prob']:
                            shadow_offset = random.randint(2, 5)
                            shadow_color = tuple(int(c * 0.3) for c in text_color) + (255,)
                            text_draw.text((10 + shadow_offset, 10 + shadow_offset), block_text, font=font, fill=shadow_color)
                        
                        # Border
                        if random.random() < effects['border_prob']:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        text_draw.text((10 + dx, 10 + dy), block_text, font=font, fill=(0, 0, 0, 255))
                        
                        # Main text
                        text_draw.text((10, 10), block_text, font=font, fill=text_color + (255,))
                        
                        # Rotate the text image
                        rotated_text = text_img.rotate(-angle, expand=True, fillcolor=(0, 0, 0, 0))
                        
                        # Calculate new position after rotation
                        rot_w, rot_h = rotated_text.size
                        paste_x = max(0, x - (rot_w - text_width) // 2)
                        paste_y = max(0, y - (rot_h - text_height) // 2)
                        
                        # Paste rotated text onto background
                        bg.paste(rotated_text, (paste_x, paste_y), rotated_text)
                        
                        # Update bounding box for rotated text
                        # Calculate the four corners of the rotated rectangle
                        import math
                        cx, cy = x + text_width/2, y + text_height/2  # center
                        cos_a = math.cos(math.radians(angle))
                        sin_a = math.sin(math.radians(angle))
                        
                        # Four corners before rotation (relative to center)
                        corners = [
                            (-text_width/2, -text_height/2),
                            (text_width/2, -text_height/2),
                            (text_width/2, text_height/2),
                            (-text_width/2, text_height/2)
                        ]
                        
                        # Rotate corners
                        rotated_corners = []
                        for dx, dy in corners:
                            rx = dx * cos_a - dy * sin_a + cx
                            ry = dx * sin_a + dy * cos_a + cy
                            rotated_corners.append([int(rx), int(ry)])
                        
                        # Create annotation with rotated polygon
                        annotation = {
                            'text': block_text,
                            'bbox': [
                                int(min(c[0] for c in rotated_corners)),
                                int(min(c[1] for c in rotated_corners)),
                                int(max(c[0] for c in rotated_corners)),
                                int(max(c[1] for c in rotated_corners))
                            ],
                            'polygon': rotated_corners  # Already a list of [x, y] pairs
                        }
                    else:
                        # No rotation - original code
                        # Shadow
                        if random.random() < effects['shadow_prob']:
                            shadow_offset = random.randint(2, 5)
                            shadow_color = tuple(int(c * 0.3) for c in text_color)
                            draw.text((x + shadow_offset, y + shadow_offset), block_text, font=font, fill=shadow_color)
                        
                        # Border
                        if random.random() < effects['border_prob']:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        draw.text((x + dx, y + dy), block_text, font=font, fill=(0, 0, 0))
                        
                        # Main text
                        draw.text((x, y), block_text, font=font, fill=text_color)
                        
                        # Create annotation
                        annotation = {
                            'text': block_text,
                            'bbox': [x, y, x + text_width, y + text_height],
                            'polygon': [
                                [x, y],
                                [x + text_width, y],
                                [x + text_width, y + text_height],
                                [x, y + text_height]
                            ]
                        }
                    
                    annotations.append(annotation)
                    break
        
        return bg, annotations
    
    def _save_lmdb(self, image: Image.Image, text: str, lmdb_env, idx: int):
        """Save cropped text image to LMDB for recognizer."""
        # Convert to bytes
        img_bytes = cv2.imencode('.png', np.array(image))[1].tobytes()
        
        # Create entry
        entry = {
            'image': img_bytes,
            'text': text,
            'idx': idx
        }
        
        # Save to LMDB
        with lmdb_env.begin(write=True) as txn:
            txn.put(str(idx).encode(), pickle.dumps(entry))
    
    def generate(self):
        """Generate complete dataset."""
        self.logger.info(f"Starting dataset generation with {self.num_images} images")
        
        # Calculate split sizes
        train_size = int(self.num_images * self.split_ratio[0])
        val_size = int(self.num_images * self.split_ratio[1])
        test_size = self.num_images - train_size - val_size
        
        splits = {
            'train': train_size,
            'val': val_size,
            'test': test_size
        }
        
        global_idx = 0
        
        for split_name, split_size in splits.items():
            self.logger.info(f"Generating {split_name} split with {split_size} images")
            
            # Setup directories
            split_dir = ensure_dir(self.output_dir / split_name)
            images_dir = ensure_dir(split_dir / 'images')
            lmdb_dir = ensure_dir(split_dir / 'lmdb')
            
            # Setup LMDB
            lmdb_env = lmdb.open(str(lmdb_dir), map_size=50 * 1024 * 1024 * 1024)  # 50GB
            
            # Annotations
            annotations = {
                'images': [],
                'annotations': [],
                'categories': [{'id': 1, 'name': 'text'}]
            }
            
            for i in tqdm(range(split_size), desc=f"Generating {split_name}"):
                # Generate text
                text = self._generate_text()
                
                # Sample font
                font_path = self._sample_font()
                
                # Render text on image
                image, text_annotations = self._render_text_on_image(
                    text, font_path, (self.resolution, self.resolution)
                )
                
                # Apply augmentations
                image_np = np.array(image)
                augmented = self.augmentations(image=image_np)
                image_aug = Image.fromarray(augmented['image'])
                
                # Save detection image
                image_name = f"{split_name}_{i:06d}.png"
                image_path = images_dir / image_name
                image_aug.save(image_path)
                
                # Add to COCO annotations
                image_info = {
                    'id': global_idx,
                    'file_name': image_name,
                    'width': self.resolution,
                    'height': self.resolution
                }
                annotations['images'].append(image_info)
                
                # Process each text instance
                for ann_idx, ann in enumerate(text_annotations):
                    # Crop text region for recognizer
                    bbox = ann['bbox']
                    text_crop = image_aug.crop(bbox)
                    
                    # Save to LMDB
                    self._save_lmdb(text_crop, ann['text'], lmdb_env, global_idx)
                    
                    # Add detection annotation
                    x, y, x2, y2 = bbox
                    w, h = x2 - x, y2 - y
                    
                    det_ann = {
                        'id': global_idx * 100 + ann_idx,
                        'image_id': global_idx,
                        'category_id': 1,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'segmentation': [[coord for point in ann['polygon'] for coord in point]],  # Flatten polygon
                        'iscrowd': 0,
                        'text': ann['text']
                    }
                    annotations['annotations'].append(det_ann)
                
                global_idx += 1
            
            # Save annotations
            ann_path = split_dir / 'annotations.json'
            with open(ann_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            # Close LMDB
            lmdb_env.close()
            
            self.logger.info(f"Completed {split_name} split")
        
        self.logger.info("Dataset generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Aurebesh dataset")
    parser.add_argument("--num_images", type=int, default=20000, help="Total images to generate")
    parser.add_argument("--output_dir", type=Path, default="data/synth", help="Output directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1], 
                       help="Train/val/test split ratio")
    parser.add_argument("--config", type=Path, help="Dataset config path")
    
    args = parser.parse_args()
    
    generator = AurebeshDatasetGenerator(
        output_dir=args.output_dir,
        num_images=args.num_images,
        resolution=args.resolution,
        split_ratio=args.split_ratio,
        config_path=args.config
    )
    
    generator.generate()


if __name__ == "__main__":
    main()