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
        
        # Probability of mixing alphabets with aurebesh
        self.alphabet_mix_prob = 0.1  # 10% chance to include alphabets
        
    def _load_fonts(self) -> Dict[str, List[Path]]:
        """Load fonts from assets/fonts directory."""
        fonts = {
            'core': list((self.font_dir / 'core').glob('*.ttf')) + 
                   list((self.font_dir / 'core').glob('*.otf')),
            'variant': list((self.font_dir / 'variant').glob('*.ttf')) + 
                      list((self.font_dir / 'variant').glob('*.otf'))
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
            ['core', 'variant'],
            weights=[probs['core_prob'], probs['variant_prob']]
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
    
    def _generate_text_with_alphabet_mix(self) -> Tuple[str, List[bool]]:
        """Generate text with mixed Aurebesh and alphabet characters.
        Returns:
            tuple: (text, is_alphabet_list) where is_alphabet_list indicates which words are alphabets
        """
        text_config = self.config['style']['text']
        num_words = random.randint(text_config['min_words'], text_config['max_words'])
        words = []
        is_alphabet_list = []
        
        for _ in range(num_words):
            # Decide if this word should be alphabet
            if random.random() < self.alphabet_mix_prob:
                # Generate alphabet word
                word_len = random.randint(
                    text_config['min_word_length'], 
                    text_config['max_word_length']
                )
                # Use standard English alphabet
                alphabet_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                word = ''.join(random.choices(alphabet_chars, k=word_len))
                words.append(word)
                is_alphabet_list.append(True)
            else:
                # Generate Aurebesh word
                if self.word_list and random.random() < (1 - DEFAULT_RANDOM_TEXT_RATIO):
                    # Use vocabulary
                    word = random.choice(self.word_list)
                else:
                    # Random Aurebesh characters
                    word_len = random.randint(
                        text_config['min_word_length'], 
                        text_config['max_word_length']
                    )
                    available_chars = [c for c in self.charset if c.isalpha()]
                    word = ''.join(random.choices(available_chars, k=word_len))
                words.append(word)
                is_alphabet_list.append(False)
        
        return ' '.join(words), is_alphabet_list
    
    def _create_spatial_grid(self, image_size: Tuple[int, int], cell_size: int = 100) -> List[List[bool]]:
        """Create a spatial grid to track occupied regions for efficient placement."""
        grid_width = (image_size[0] + cell_size - 1) // cell_size
        grid_height = (image_size[1] + cell_size - 1) // cell_size
        return [[False for _ in range(grid_width)] for _ in range(grid_height)]
    
    def _mark_grid_occupied(self, grid: List[List[bool]], bbox: List[int], cell_size: int = 100):
        """Mark grid cells as occupied by a bounding box."""
        x1, y1, x2, y2 = bbox
        start_row = max(0, y1 // cell_size)
        end_row = min(len(grid), (y2 + cell_size - 1) // cell_size)
        start_col = max(0, x1 // cell_size)
        end_col = min(len(grid[0]), (x2 + cell_size - 1) // cell_size)
        
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                grid[row][col] = True
    
    def _check_grid_available(self, grid: List[List[bool]], bbox: List[int], cell_size: int = 100) -> bool:
        """Check if grid cells for a bounding box are available."""
        x1, y1, x2, y2 = bbox
        start_row = max(0, y1 // cell_size)
        end_row = min(len(grid), (y2 + cell_size - 1) // cell_size)
        start_col = max(0, x1 // cell_size)
        end_col = min(len(grid[0]), (x2 + cell_size - 1) // cell_size)
        
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if grid[row][col]:
                    return False
        return True
    
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
    
    def _render_multiline_text(
        self,
        lines: List[str],
        font_path: Path,
        font_sizes: List[int],
        x: int,
        y: int,
        text_color: Tuple[int, int, int],
        draw: ImageDraw.Draw,
        line_spacing: float = 1.2
    ) -> List[Dict]:
        """Render multiple lines of text with individual bboxes."""
        annotations = []
        current_y = y
        effects = self.config['style']['effects']
        
        for i, (line_text, font_size) in enumerate(zip(lines, font_sizes)):
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text metrics
            bbox = draw.textbbox((0, 0), line_text, font=font)
            text_height = bbox[3] - bbox[1]
            actual_x_offset = -bbox[0]
            actual_y_offset = -bbox[1]
            
            # Adjust position
            text_x = x + actual_x_offset
            text_y = current_y + actual_y_offset
            
            # Shadow
            if random.random() < effects['shadow_prob']:
                shadow_offset = random.randint(2, 5)
                shadow_color = tuple(int(c * 0.3) for c in text_color)
                draw.text((text_x + shadow_offset, text_y + shadow_offset), 
                        line_text, font=font, fill=shadow_color)
            
            # Border
            if random.random() < effects['border_prob']:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), 
                                    line_text, font=font, fill=(0, 0, 0))
            
            # Draw the line
            draw.text((text_x, text_y), line_text, font=font, fill=text_color)
            
            # Get precise bbox for this line
            line_bbox = draw.textbbox((text_x, text_y), line_text, font=font)
            
            # Create annotation for this line
            annotation = {
                'text': line_text,
                'bbox': [line_bbox[0], line_bbox[1], line_bbox[2], line_bbox[3]],
                'polygon': [
                    [line_bbox[0], line_bbox[1]],
                    [line_bbox[2], line_bbox[1]],
                    [line_bbox[2], line_bbox[3]],
                    [line_bbox[0], line_bbox[3]]
                ]
            }
            annotations.append(annotation)
            
            # Move to next line
            current_y += int(text_height * line_spacing)
        
        return annotations
    
    def _render_mixed_text(
        self,
        text: str,
        is_alphabet_list: List[bool],
        font_path: Path,
        font_size: int,
        x: int,
        y: int,
        text_color: Tuple[int, int, int],
        draw: ImageDraw.Draw
    ) -> List[Dict]:
        """Render text with mixed Aurebesh and alphabet characters.
        Only Aurebesh words get bounding boxes."""
        words = text.split()
        annotations = []
        current_x = x
        effects = self.config['style']['effects']
        
        for word, is_alphabet in zip(words, is_alphabet_list):
            # Choose font based on character type
            if is_alphabet:
                # Use default system font for alphabets
                try:
                    # Try to use a clean sans-serif font
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            else:
                # Use Aurebesh font
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except:
                    font = ImageFont.load_default()
            
            # Get text metrics
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            actual_x_offset = -bbox[0]
            actual_y_offset = -bbox[1]
            
            # Adjust position
            word_x = current_x + actual_x_offset
            word_y = y + actual_y_offset
            
            # Apply effects (shadow/border) for all text
            if random.random() < effects['shadow_prob']:
                shadow_offset = random.randint(2, 5)
                shadow_color = tuple(int(c * 0.3) for c in text_color)
                draw.text((word_x + shadow_offset, word_y + shadow_offset), 
                        word, font=font, fill=shadow_color)
            
            if random.random() < effects['border_prob']:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((word_x + dx, word_y + dy), 
                                    word, font=font, fill=(0, 0, 0))
            
            # Draw the word
            draw.text((word_x, word_y), word, font=font, fill=text_color)
            
            # Only create annotation for Aurebesh words
            if not is_alphabet:
                word_bbox = draw.textbbox((word_x, word_y), word, font=font)
                annotation = {
                    'text': word,
                    'bbox': [word_bbox[0], word_bbox[1], word_bbox[2], word_bbox[3]],
                    'polygon': [
                        [word_bbox[0], word_bbox[1]],
                        [word_bbox[2], word_bbox[1]],
                        [word_bbox[2], word_bbox[3]],
                        [word_bbox[0], word_bbox[3]]
                    ]
                }
                annotations.append(annotation)
            
            # Move to next word position (add space)
            space_width = draw.textbbox((0, 0), ' ', font=font)[2]
            current_x += text_width + space_width
        
        return annotations
    
    def _generate_text_variant(self) -> Tuple[str, str]:
        """Generate text and determine variant type."""
        # Decide variant type based on probabilities
        variant_config = self.config['style'].get('variants', {})
        multiline_prob = variant_config.get('multiline_prob', 0.3)
        size_contrast_prob = variant_config.get('size_contrast_prob', 0.2)
        
        rand = random.random()
        if rand < multiline_prob:
            return self._generate_text(), 'multiline'
        elif rand < multiline_prob + size_contrast_prob:
            return self._generate_text(), 'size_contrast'
        else:
            return self._generate_text(), 'normal'
    
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
        
        # Generate multiple text blocks based on config
        layout_config = self.config['style']['layout']
        max_blocks = layout_config.get('max_text_blocks', 10)
        num_text_blocks = random.randint(1, max_blocks)
        annotations = []
        occupied_regions = []  # To avoid overlapping text
        
        # Adaptive spacing based on number of blocks
        base_min_spacing = layout_config.get('min_text_spacing', 30)
        adaptive_spacing = layout_config.get('adaptive_spacing', True)
        if adaptive_spacing and num_text_blocks > 5:
            # Reduce spacing for more blocks, but keep a minimum
            spacing_factor = max(0.5, 1.0 - (num_text_blocks - 5) * 0.1)
            min_spacing = max(20, int(base_min_spacing * spacing_factor))
        else:
            min_spacing = base_min_spacing
        
        # Create spatial grid for efficient placement tracking
        # Use finer grid for more blocks
        grid_cell_size = 30 if num_text_blocks > 5 else 50
        spatial_grid = self._create_spatial_grid(image_size, cell_size=grid_cell_size)
        
        for block_idx in range(num_text_blocks):
            # Decide if this block should have mixed alphabet
            use_alphabet_mix = random.random() < self.alphabet_mix_prob
            is_alphabet_list = []
            
            # Generate text and variant type for this block
            if block_idx == 0:
                if use_alphabet_mix:
                    block_text, is_alphabet_list = self._generate_text_with_alphabet_mix()
                    variant_type = 'mixed'
                else:
                    block_text = text
                    variant_type = self._generate_text_variant()[1]
            else:
                if use_alphabet_mix:
                    block_text, is_alphabet_list = self._generate_text_with_alphabet_mix()
                    variant_type = 'mixed'
                else:
                    block_text, variant_type = self._generate_text_variant()
            
            # Sample font for this block (can vary per block)
            if block_idx > 0:
                font_path = self._sample_font()
            
            # Setup font with varying sizes - adapt based on number of blocks
            if num_text_blocks > 7:
                # Many blocks - use smaller fonts
                max_font_size = 50
                min_font_size = 20
            elif num_text_blocks > 4:
                # Medium number of blocks
                max_font_size = 65
                min_font_size = 25
            else:
                # Few blocks - can use larger fonts
                max_font_size = 80
                min_font_size = 30
            
            font_size = random.randint(min_font_size, max_font_size)
            
            # Try to find a font size that fits with proper margins
            margin = 150  # Increased margin for safety to prevent text cutoff after augmentation
            max_rotation_angle = 15  # Maximum rotation in degrees
            
            for size_attempt in range(10):  # More attempts to find suitable font size
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except:
                    self.logger.warning(f"Failed to load font {font_path}, using default")
                    font = ImageFont.load_default()
                
                # Get accurate text metrics using textbbox
                if variant_type == 'mixed':
                    # For mixed text, calculate width for each word separately
                    words = block_text.split()
                    total_width = 0
                    max_height = 0
                    
                    for word, is_alphabet in zip(words, is_alphabet_list):
                        if is_alphabet:
                            # Use system font for alphabet
                            try:
                                word_font = ImageFont.truetype("Arial.ttf", font_size)
                            except:
                                word_font = ImageFont.load_default()
                        else:
                            word_font = font
                        
                        word_bbox = draw.textbbox((0, 0), word, font=word_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        word_height = word_bbox[3] - word_bbox[1]
                        total_width += word_width
                        max_height = max(max_height, word_height)
                    
                    # Add space between words
                    space_bbox = draw.textbbox((0, 0), ' ', font=font)
                    space_width = space_bbox[2]
                    text_width = total_width + space_width * (len(words) - 1)
                    text_height = max_height
                else:
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
            
            # Handle different text variants
            if variant_type == 'multiline':
                # Split text into 2-3 lines
                words = block_text.split()
                num_lines = random.randint(2, 3)
                
                # Distribute words across lines
                words_per_line = max(1, len(words) // num_lines)
                lines = []
                for i in range(num_lines):
                    start_idx = i * words_per_line
                    if i == num_lines - 1:
                        # Last line gets all remaining words
                        line_words = words[start_idx:]
                    else:
                        line_words = words[start_idx:start_idx + words_per_line]
                    if line_words:
                        lines.append(' '.join(line_words))
                
                # All lines use same font size
                font_sizes = [font_size] * len(lines)
                
            elif variant_type == 'size_contrast':
                # Two lines with 2x size difference
                words = block_text.split()
                if len(words) >= 2:
                    # Split into two lines
                    mid = len(words) // 2
                    lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
                    # First line is 2x larger
                    font_sizes = [font_size, max(15, font_size // 2)]
                else:
                    # Single word - duplicate it
                    lines = [words[0], words[0] if words else block_text]
                    font_sizes = [font_size, max(15, font_size // 2)]
            else:
                # Normal single line
                lines = [block_text]
                font_sizes = [font_size]
            
            # Decide on rotation first to calculate accurate dimensions
            effects = self.config['style']['effects']
            will_rotate = 'rotation_prob' in effects and random.random() < effects['rotation_prob'] and variant_type == 'normal'
            # Calculate dimensions for all lines
            if variant_type in ['multiline', 'size_contrast']:
                # Calculate total height and max width for multi-line text
                total_height = 0
                max_width = 0
                line_spacing = 1.2
                
                for line_text, line_font_size in zip(lines, font_sizes):
                    try:
                        line_font = ImageFont.truetype(str(font_path), line_font_size)
                    except:
                        line_font = ImageFont.load_default()
                    
                    bbox = draw.textbbox((0, 0), line_text, font=line_font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                    
                    max_width = max(max_width, line_width)
                    total_height += line_height * line_spacing
                
                placement_width = max_width
                placement_height = total_height
                angle = 0  # No rotation for multi-line text
            elif will_rotate:
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
                
                # Check overlap with existing text blocks using adaptive spacing
                padding = min_spacing  # Use the adaptive spacing calculated earlier
                new_region = [
                    x - padding, 
                    y - padding, 
                    x + int(placement_width) + padding, 
                    y + int(placement_height) + padding
                ]
                
                # First check spatial grid for quick rejection
                grid_bbox = [x, y, x + int(placement_width), y + int(placement_height)]
                if not self._check_grid_available(spatial_grid, grid_bbox, grid_cell_size):
                    continue
                
                # Then check detailed overlap with padding
                overlap = False
                for region in occupied_regions:
                    if (new_region[0] < region[2] and new_region[2] > region[0] and
                        new_region[1] < region[3] and new_region[3] > region[1]):
                        overlap = True
                        break
                
                if not overlap or attempt == max_attempts - 1:
                    # Found non-overlapping position or last attempt
                    occupied_regions.append(new_region)
                    
                    # Mark spatial grid as occupied
                    self._mark_grid_occupied(spatial_grid, new_region, grid_cell_size)
                    
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
                            annotations.append(annotation)
                        else:
                            # Skip if no visible pixels
                            continue
                    else:
                        # No rotation - handle single or multi-line text
                        if variant_type == 'mixed':
                            # Use the mixed text rendering function
                            mixed_annotations = self._render_mixed_text(
                                block_text, is_alphabet_list, font_path, font_size, 
                                x, y, text_color, draw
                            )
                            
                            # Filter out annotations that exceed bounds
                            valid_annotations = []
                            for ann in mixed_annotations:
                                bbox = ann['bbox']
                                if not (bbox[0] < 0 or bbox[1] < 0 or
                                        bbox[2] > image_size[0] or bbox[3] > image_size[1]):
                                    valid_annotations.append(ann)
                            
                            # Only add if we have at least one valid annotation
                            if valid_annotations:
                                annotations.extend(valid_annotations)
                        elif variant_type in ['multiline', 'size_contrast']:
                            # Use the multiline rendering function
                            line_annotations = self._render_multiline_text(
                                lines, font_path, font_sizes, x, y, text_color, draw
                            )
                            
                            # Filter out annotations that exceed bounds
                            valid_annotations = []
                            for ann in line_annotations:
                                bbox = ann['bbox']
                                if not (bbox[0] < 0 or bbox[1] < 0 or
                                        bbox[2] > image_size[0] or bbox[3] > image_size[1]):
                                    valid_annotations.append(ann)
                            
                            # Only add if we have at least one valid annotation
                            if valid_annotations:
                                annotations.extend(valid_annotations)
                        else:
                            # Single line text - original logic
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
                    valid_annotations = []
                    
                    for i, ann in enumerate(text_annotations):
                        # Convert from [x1, y1, x2, y2] to pascal_voc format [x_min, y_min, x_max, y_max]
                        bbox = ann['bbox']
                        
                        # Skip degenerate bboxes (width or height <= 0)
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            self.logger.warning(f"Skipping degenerate bbox: {bbox}")
                            continue
                        
                        # Also skip very small bboxes that might cause issues
                        min_size = 5  # Minimum width/height in pixels
                        if (bbox[2] - bbox[0] < min_size) or (bbox[3] - bbox[1] < min_size):
                            self.logger.warning(f"Skipping too small bbox: {bbox}")
                            continue
                        
                        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                        labels.append(len(valid_annotations))  # Use valid annotation index
                        valid_annotations.append(ann)
                    
                    # Skip if no valid bboxes remain
                    if not bboxes:
                        self.logger.warning("No valid bboxes after filtering, retrying...")
                        continue
                    
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
                        orig_ann = valid_annotations[label_idx]
                        
                        # Convert bbox coordinates to integers
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        
                        # Skip if bbox is outside image boundaries (text was pushed out by augmentation)
                        # Also skip if too close to edge to prevent text cutoff
                        safety_margin = 10
                        if (x1 < 0 or y1 < 0 or x2 > self.resolution or y2 > self.resolution or
                            x2 <= x1 or y2 <= y1 or
                            x1 < safety_margin or y1 < safety_margin or 
                            x2 > self.resolution - safety_margin or y2 > self.resolution - safety_margin):
                            continue  # Skip augmented text that's outside bounds or too close to edge
                        
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
                        
                        # Skip if bbox is outside image boundaries
                        # Add safety margin to prevent text cutoff at boundaries
                        safety_margin = 10
                        if (bbox[0] < 0 or bbox[1] < 0 or 
                            bbox[2] > self.resolution or bbox[3] > self.resolution):
                            self.logger.debug(f"Skipping out-of-bounds text during crop: {ann['text']}")
                            continue
                        
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