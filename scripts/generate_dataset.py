#!/usr/bin/env python3
"""
Generate synthetic Aurebesh dataset using SynthTIGER.
Creates images with text detection annotations and LMDB for recognition.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import albumentations as A
from tqdm import tqdm
import lmdb
import pickle
from wordfreq import top_n_list

from utils import setup_logger, ensure_dir, load_config, get_charset, STAR_WARS_VOCABULARY

# Default configuration
DEFAULT_WORDFREQ_LIMIT = 10000  # Top N words from wordfreq to use
DEFAULT_RANDOM_TEXT_RATIO = 0.05  # 5% random text for robustness, 95% vocabulary text


class AurebeshDatasetGenerator:
    def __init__(
        self,
        output_dir: Path,
        num_images: int = 20000,
        resolution: int = 1024,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        config_path: Optional[Path] = None,
        use_wordfreq: bool = True,
        wordfreq_limit: int = DEFAULT_WORDFREQ_LIMIT,
        custom_words: Optional[List[str]] = None,
        debug: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.resolution = resolution
        self.split_ratio = split_ratio
        self.use_wordfreq = use_wordfreq
        self.wordfreq_limit = wordfreq_limit
        self.debug = debug
        
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
        
        # Setup word vocabulary
        self.word_list = self._setup_vocabulary(custom_words)
        
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
        """Setup albumentations pipeline with bbox transformation support."""
        aug_config = self.config['augmentation']
        
        transforms = []
        
        # Perspective transform - this can change bbox coordinates
        if aug_config['perspective']['prob'] > 0:
            deg_range = aug_config['perspective']['degree_range']
            transforms.append(
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=aug_config['perspective']['prob']
                )
            )
        
        # Noise - doesn't affect bboxes
        if aug_config['noise']['prob'] > 0:
            transforms.append(
                A.GaussNoise(
                    std_range=aug_config['noise']['var_range'],
                    p=aug_config['noise']['prob']
                )
            )
        
        # Blur - doesn't affect bboxes
        if aug_config['blur']['prob'] > 0:
            kernel_range = aug_config['blur']['kernel_range']
            # Ensure minimum kernel size is 3
            kernel_range = [max(3, kernel_range[0]), max(3, kernel_range[1])]
            transforms.append(
                A.Blur(
                    blur_limit=kernel_range,
                    p=aug_config['blur']['prob']
                )
            )
        
        # JPEG compression - doesn't affect bboxes
        if aug_config['jpeg']['prob'] > 0:
            quality_range = aug_config['jpeg']['quality_range']
            transforms.append(
                A.ImageCompression(
                    quality_range=quality_range,
                    p=aug_config['jpeg']['prob']
                )
            )
        
        # Configure bbox parameters for albumentations
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['labels'],
            min_visibility=0.3  # Keep bbox if at least 30% is visible after transformation
        ))
    
    def _setup_vocabulary(self, custom_words: Optional[List[str]] = None) -> List[str]:
        """Setup word vocabulary from wordfreq and custom words."""
        vocabulary = []
        wordfreq_loaded_words = []

        if self.use_wordfreq:
            # Get top words in English
            wordfreq_loaded_words = top_n_list('en', self.wordfreq_limit)
            wordfreq_loaded_words = [word.upper() for word in wordfreq_loaded_words]

        # Add custom words (Star Wars themed)
        if custom_words is None:
            custom_words = STAR_WARS_VOCABULARY
        custom_words = [word.upper() for word in custom_words if word.isalpha()]

        # Merge with vocabulary
        vocabulary = wordfreq_loaded_words + custom_words

        # Remove duplicates by converting to set
        vocabulary = list(set(vocabulary))

        self.logger.info(f"Added {len(custom_words)} custom words (excluded {len(wordfreq_loaded_words) + len(custom_words) - len(vocabulary)} duplicates)")
        self.logger.info(f"Total vocabulary size: {len(vocabulary)} words")
        
        return vocabulary
    
    def _sample_font(self) -> Path:
        """Sample font based on configured probabilities."""
        probs = self.config['style']['font']
        category = random.choices(
            ['core', 'variant', 'fancy'],
            weights=[probs['core_prob'], probs['variant_prob'], probs['fancy_prob']]
        )[0]
        return random.choice(self.fonts[category])
    
    def _generate_text(self) -> str:
        """Generate text using vocabulary or random characters."""
        text_config = self.config['style']['text']
        num_words = random.randint(text_config['min_words'], text_config['max_words'])
        
        if self.word_list and random.random() < (1 - DEFAULT_RANDOM_TEXT_RATIO):  # 95% use vocabulary
            # Use words from vocabulary
            words = random.choices(self.word_list, k=num_words)
            return ' '.join(words)
        else:
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
        
        # Generate multiple text blocks (1-10 per image)
        num_text_blocks = random.randint(1, 10)
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
            
            # Setup font with varying sizes - try to fit text in image
            max_font_size = 80
            min_font_size = 20
            font_size = random.randint(min_font_size, max_font_size)
            
            # Try to find a font size that fits with proper margins
            margin = 50  # Increased margin for safety
            max_rotation_angle = 15  # Maximum rotation in degrees
            
            for size_attempt in range(10):  # More attempts to find suitable font size
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except:
                    self.logger.warning(f"Failed to load font {font_path}, using default")
                    font = ImageFont.load_default()
                
                # Get accurate text metrics using textbbox
                bbox = draw.textbbox((0, 0), block_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Calculate maximum dimensions considering potential rotation
                import math
                angle_rad = math.radians(max_rotation_angle)
                cos_a = abs(math.cos(angle_rad))
                sin_a = abs(math.sin(angle_rad))
                
                # Rotated bounding box dimensions (worst case)
                rotated_width = text_width * cos_a + text_height * sin_a
                rotated_height = text_width * sin_a + text_height * cos_a
                
                # Check if text fits in image with generous margins
                if (rotated_width <= image_size[0] - 2 * margin and 
                    rotated_height <= image_size[1] - 2 * margin):
                    break  # Font size works
                
                # Reduce font size for next attempt
                font_size = max(min_font_size, int(font_size * 0.85))
            
            # Skip this text block if still too large
            if (rotated_width > image_size[0] - 2 * margin or 
                rotated_height > image_size[1] - 2 * margin):
                continue
            
            # Text color for this block
            text_color = tuple(random.randint(0, 255) for _ in range(3))
            
            # Decide on rotation first to calculate accurate dimensions
            effects = self.config['style']['effects']
            will_rotate = 'rotation_prob' in effects and random.random() < effects['rotation_prob']
            if will_rotate:
                rotation_range = effects.get('rotation_range', [-15, 15])
                angle = random.uniform(rotation_range[0], rotation_range[1])
                # Use rotated dimensions for placement with extra safety margin
                # Add 20% extra margin for rotation safety
                safety_factor = 1.2
                placement_width = rotated_width * safety_factor
                placement_height = rotated_height * safety_factor
            else:
                angle = 0
                # Use original text dimensions for placement
                placement_width = text_width
                placement_height = text_height
            
            # Try to find non-overlapping position with accurate dimensions
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Calculate safe placement boundaries with accurate dimensions
                safe_margin = margin
                max_x = image_size[0] - int(placement_width) - safe_margin
                max_y = image_size[1] - int(placement_height) - safe_margin
                
                if max_x <= safe_margin or max_y <= safe_margin:
                    break  # Text too large for image
                
                # Position text with safe margins
                x = random.randint(safe_margin, max_x)
                y = random.randint(safe_margin, max_y)
                
                # Check overlap with existing text blocks using larger padding
                padding = 15
                new_region = [
                    x - padding, 
                    y - padding, 
                    x + int(placement_width) + padding, 
                    y + int(placement_height) + padding
                ]
                
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
                    
                    # Draw text with pre-determined rotation
                    if will_rotate:
                        # Create text image with precise dimensions
                        # Add padding to prevent clipping during rotation
                        padding = 20
                        temp_size = (int(text_width) + 2*padding, int(text_height) + 2*padding)
                        text_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
                        text_draw = ImageDraw.Draw(text_img)
                        
                        # Center text in temporary image
                        temp_x = padding
                        temp_y = padding
                        
                        # Draw text with effects on temporary image
                        # Shadow
                        if random.random() < effects['shadow_prob']:
                            shadow_offset = random.randint(2, 5)
                            shadow_color = tuple(int(c * 0.3) for c in text_color) + (255,)
                            text_draw.text((temp_x + shadow_offset, temp_y + shadow_offset), 
                                         block_text, font=font, fill=shadow_color)
                        
                        # Border
                        if random.random() < effects['border_prob']:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        text_draw.text((temp_x + dx, temp_y + dy), 
                                                     block_text, font=font, fill=(0, 0, 0, 255))
                        
                        # Main text
                        text_draw.text((temp_x, temp_y), block_text, font=font, fill=text_color + (255,))
                        
                        # Rotate the text image
                        rotated_text = text_img.rotate(-angle, expand=True, fillcolor=(0, 0, 0, 0))
                        
                        # Calculate precise paste position
                        rot_w, rot_h = rotated_text.size
                        
                        # Center the rotated text within the allocated region
                        paste_x = x + int(placement_width - rot_w) // 2
                        paste_y = y + int(placement_height - rot_h) // 2
                        
                        # Final boundary check - if rotated text would exceed bounds, skip this text
                        if (paste_x < 0 or paste_y < 0 or 
                            paste_x + rot_w > image_size[0] or 
                            paste_y + rot_h > image_size[1]):
                            continue  # Skip this text block
                        
                        # Paste rotated text onto background
                        bg.paste(rotated_text, (paste_x, paste_y), rotated_text)
                        
                        # Find actual non-transparent pixels to create accurate bbox
                        # Convert rotated text to numpy for easier processing
                        rot_array = np.array(rotated_text)
                        alpha_channel = rot_array[:, :, 3]  # Alpha channel
                        
                        # Find bounds of non-transparent pixels
                        y_indices, x_indices = np.where(alpha_channel > 0)
                        
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            # Calculate actual text bounds relative to paste position
                            actual_x1 = paste_x + np.min(x_indices)
                            actual_y1 = paste_y + np.min(y_indices)
                            actual_x2 = paste_x + np.max(x_indices) + 1
                            actual_y2 = paste_y + np.max(y_indices) + 1
                            
                            # Check if text bounds are within image (skip if outside)
                            if (actual_x1 < 0 or actual_y1 < 0 or 
                                actual_x2 > image_size[0] or actual_y2 > image_size[1]):
                                continue  # Skip this text block if it extends outside image bounds
                            
                            # Create annotation with actual bounds (no clamping)
                            annotation = {
                                'text': block_text,
                                'bbox': [actual_x1, actual_y1, actual_x2, actual_y2],
                                'polygon': [
                                    [actual_x1, actual_y1],
                                    [actual_x2, actual_y1],
                                    [actual_x2, actual_y2],
                                    [actual_x1, actual_y2]
                                ]
                            }
                        else:
                            # Skip if no visible pixels
                            continue
                    else:
                        # No rotation - improved non-rotated text placement
                        # Calculate precise text position with baseline considerations
                        # Get accurate text metrics
                        bbox = draw.textbbox((0, 0), block_text, font=font)
                        actual_x_offset = -bbox[0]  # Left bearing
                        actual_y_offset = -bbox[1]  # Top bearing
                        
                        # Adjust position to account for font metrics
                        text_x = x + actual_x_offset
                        text_y = y + actual_y_offset
                        
                        # Final boundary check - ensure text won't exceed image bounds
                        final_bbox_check = draw.textbbox((text_x, text_y), block_text, font=font)
                        if (final_bbox_check[0] < 0 or final_bbox_check[1] < 0 or
                            final_bbox_check[2] > image_size[0] or final_bbox_check[3] > image_size[1]):
                            continue  # Skip this text block
                        
                        # Shadow
                        if random.random() < effects['shadow_prob']:
                            shadow_offset = random.randint(2, 5)
                            shadow_color = tuple(int(c * 0.3) for c in text_color)
                            draw.text((text_x + shadow_offset, text_y + shadow_offset), 
                                    block_text, font=font, fill=shadow_color)
                        
                        # Border
                        if random.random() < effects['border_prob']:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        draw.text((text_x + dx, text_y + dy), 
                                                block_text, font=font, fill=(0, 0, 0))
                        
                        # Main text
                        draw.text((text_x, text_y), block_text, font=font, fill=text_color)
                        
                        # Create precise bounding box using actual text metrics
                        final_bbox = draw.textbbox((text_x, text_y), block_text, font=font)
                        
                        # Create annotation with precise coordinates (no clamping needed)
                        annotation = {
                            'text': block_text,
                            'bbox': [final_bbox[0], final_bbox[1], final_bbox[2], final_bbox[3]],
                            'polygon': [
                                [final_bbox[0], final_bbox[1]],
                                [final_bbox[2], final_bbox[1]],
                                [final_bbox[2], final_bbox[3]],
                                [final_bbox[0], final_bbox[3]]
                            ]
                        }
                    
                    annotations.append(annotation)
                    break
        
        return bg, annotations
    
    def _save_debug_image(self, image: Image.Image, annotations: List[Dict], output_path: Path):
        """Save image with bounding boxes drawn for debugging."""
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        
        for ann in annotations:
            bbox = ann['bbox']
            polygon = ann['polygon']
            
            # Draw bounding box in red
            draw.rectangle(bbox, outline='red', width=2)
            
            # Draw polygon in blue
            if len(polygon) >= 3:
                flat_polygon = [coord for point in polygon for coord in point]
                draw.polygon(flat_polygon, outline='blue', width=2)
            
            # Draw text label
            text_pos = (bbox[0], bbox[1] - 20)
            draw.text(text_pos, ann['text'][:20], fill='yellow')
        
        debug_img.save(output_path)

    def _save_lmdb(self, image: Image.Image, text: str, lmdb_env, lmdb_idx: int):
        """Save cropped text image to LMDB for recognizer."""
        # Convert to bytes
        img_bytes = cv2.imencode('.png', np.array(image))[1].tobytes()
        
        # Create entry
        entry = {
            'image': img_bytes,
            'text': text,
            'idx': lmdb_idx
        }
        
        # Save to LMDB
        with lmdb_env.begin(write=True) as txn:
            txn.put(str(lmdb_idx).encode(), pickle.dumps(entry))
    
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
        global_lmdb_idx = 0  # Separate counter for LMDB entries
        
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
            
            generated_count = 0
            attempts = 0
            max_attempts = split_size * 2  # Allow up to 2x attempts to get required images
            
            with tqdm(total=split_size, desc=f"Generating {split_name}") as pbar:
                while generated_count < split_size and attempts < max_attempts:
                    # Generate text
                    text = self._generate_text()
                    
                    # Sample font
                    font_path = self._sample_font()
                    
                    # Render text on image
                    image, text_annotations = self._render_text_on_image(
                        text, font_path, (self.resolution, self.resolution)
                    )
                    
                    attempts += 1
                    
                    # Skip images with no text annotations
                    if not text_annotations:
                        self.logger.warning(f"No text placed on image attempt {attempts}, retrying...")
                        continue
                    
                    # Apply augmentations with bbox transformation
                    image_np = np.array(image)
                    
                    # Prepare bboxes for albumentations (convert to pascal_voc format)
                    bboxes = []
                    labels = []
                    for i, ann in enumerate(text_annotations):
                        # Convert from [x1, y1, x2, y2] to pascal_voc format [x_min, y_min, x_max, y_max]
                        bbox = ann['bbox']
                        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                        labels.append(i)  # Use index as label
                    
                    # Apply augmentations
                    augmented = self.augmentations(
                        image=image_np,
                        bboxes=bboxes,
                        labels=labels
                    )
                    
                    image_aug = Image.fromarray(augmented['image'])
                    augmented_bboxes = augmented['bboxes']
                    augmented_labels = augmented['labels']
                    
                    # Update text_annotations with transformed bboxes
                    updated_annotations = []
                    for i, (bbox, label) in enumerate(zip(augmented_bboxes, augmented_labels)):
                        # Get original annotation (ensure label is integer)
                        label_idx = int(label)
                        orig_ann = text_annotations[label_idx]
                        
                        # Convert bbox coordinates to integers
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        
                        # Skip if bbox is outside image boundaries (text was pushed out by augmentation)
                        if (x1 < 0 or y1 < 0 or x2 > self.resolution or y2 > self.resolution or
                            x2 <= x1 or y2 <= y1):
                            continue  # Skip augmented text that's outside bounds
                        
                        # Update bbox coordinates (no clamping)
                        updated_ann = orig_ann.copy()
                        updated_ann['bbox'] = [x1, y1, x2, y2]
                        
                        # Update polygon
                        updated_ann['polygon'] = [
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                        ]
                        
                        updated_annotations.append(updated_ann)
                    
                    # Use updated annotations for saving
                    text_annotations = updated_annotations
                    
                    # Save detection image
                    image_name = f"{split_name}_{generated_count:06d}.png"
                    image_path = images_dir / image_name
                    image_aug.save(image_path)
                    
                    # Save debug image with bboxes (if debug mode enabled)
                    if self.debug and generated_count < 20:  # Save debug for first 20 images
                        debug_dir = ensure_dir(split_dir / 'debug')
                        debug_name = f"{split_name}_{generated_count:06d}_debug.png"
                        debug_path = debug_dir / debug_name
                        self._save_debug_image(image_aug, text_annotations, debug_path)
                    
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
                        
                        # Save to LMDB with unique index
                        self._save_lmdb(text_crop, ann['text'], lmdb_env, global_lmdb_idx)
                        global_lmdb_idx += 1
                        
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
                    
                    generated_count += 1
                    global_idx += 1
                    pbar.update(1)
                
                if generated_count < split_size:
                    self.logger.warning(f"Could only generate {generated_count}/{split_size} images after {attempts} attempts")
            
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
    parser.add_argument("--no_wordfreq", action="store_true", help="Disable wordfreq vocabulary")
    parser.add_argument("--wordfreq_limit", type=int, default=DEFAULT_WORDFREQ_LIMIT, help="Number of wordfreq words to use")
    parser.add_argument("--custom_words", nargs="+", help="Additional custom words to add")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with bbox visualization")
    
    args = parser.parse_args()
    
    generator = AurebeshDatasetGenerator(
        output_dir=args.output_dir,
        num_images=args.num_images,
        resolution=args.resolution,
        split_ratio=args.split_ratio,
        config_path=args.config,
        use_wordfreq=not args.no_wordfreq,
        wordfreq_limit=args.wordfreq_limit,
        custom_words=args.custom_words,
        debug=args.debug
    )
    
    generator.generate()


if __name__ == "__main__":
    main()