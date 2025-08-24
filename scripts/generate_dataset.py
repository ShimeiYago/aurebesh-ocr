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
from wordfreq import top_n_list

from utils import setup_logger, ensure_dir, load_config, get_charset, STAR_WARS_VOCABULARY


DEFAULT_WORDFREQ_LIMIT = 10000


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
        self.debug = debug
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "dataset.yaml"
        self.config = load_config(config_path)
        
        # Load text generation settings from config
        text_gen_config = self.config.get('text_generation', {})
        self.actual_wordfreq_limit = text_gen_config.get('wordfreq_limit', DEFAULT_WORDFREQ_LIMIT)
        self.random_text_ratio = text_gen_config.get('random_text_ratio', 0.05)
        self.numeric_text_ratio = text_gen_config.get('numeric_text_ratio', 0.15)
        self.alphabet_mix_prob = text_gen_config.get('alphabet_mix_prob', 0.1)  # 10% default
        
        # Override wordfreq_limit if provided via constructor
        if wordfreq_limit != DEFAULT_WORDFREQ_LIMIT:
            self.actual_wordfreq_limit = wordfreq_limit
        
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
        
        # Create charset set once for filtering both wordfreq and custom words
        charset_set = set(self.charset)

        if self.use_wordfreq:
            # Get top words in English using config value
            wordfreq_loaded_words = top_n_list('en', self.actual_wordfreq_limit)
            wordfreq_loaded_words = [word.upper() for word in wordfreq_loaded_words]
            
            # Filter wordfreq words to only include those with characters in our charset
            original_count = len(wordfreq_loaded_words)
            wordfreq_loaded_words = [
                word for word in wordfreq_loaded_words 
                if all(char in charset_set for char in word)
            ]
            filtered_count = original_count - len(wordfreq_loaded_words)
            
            if filtered_count > 0:
                self.logger.info(f"Filtered out {filtered_count} wordfreq words containing invalid characters")
                self.logger.info(f"Remaining wordfreq words: {len(wordfreq_loaded_words)}")

        # Add custom words (Star Wars themed)
        if custom_words is None:
            custom_words = STAR_WARS_VOCABULARY
        custom_words = [word.upper() for word in custom_words if word.isalpha()]
        
        # Filter custom words for consistency (reuse charset_set from above)
        original_custom_count = len(custom_words)
        custom_words = [
            word for word in custom_words 
            if all(char in charset_set for char in word)
        ]
        filtered_custom_count = original_custom_count - len(custom_words)
        
        if filtered_custom_count > 0:
            self.logger.info(f"Filtered out {filtered_custom_count} custom words containing invalid characters")

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
        
        # 1% chance to generate empty text for robustness
        if random.random() < 0.01:
            return ""
        
        num_words = random.randint(text_config['min_words'], text_config['max_words'])
        
        # New probability distribution for better number coverage
        rand_val = random.random()
        
        if rand_val < (1 - self.random_text_ratio - self.numeric_text_ratio):  # ~80% vocabulary
            # Use words from vocabulary (mostly alphabetic)
            words = random.choices(self.word_list, k=num_words)
            return ' '.join(words)
        elif rand_val < (1 - self.random_text_ratio):  # ~15% numeric/mixed content
            # Generate numeric-heavy or mixed content
            return self._generate_numeric_mixed_text(num_words, text_config)
        else:  # ~5% pure random
            # Pure random characters
            words = []
            for _ in range(num_words):
                word_len = random.randint(
                    text_config['min_word_length'], 
                    text_config['max_word_length']
                )
                # Use all characters from charset for mixed content
                available_chars = list(self.charset)
                word = ''.join(random.choices(available_chars, k=word_len))
                words.append(word)
            
            return ' '.join(words)
    
    def _generate_numeric_mixed_text(self, num_words: int, text_config: dict) -> str:
        """Generate text with emphasis on numbers and realistic numeric patterns."""
        words = []
        digits = [c for c in self.charset if c.isdigit()]
        letters = [c for c in self.charset if c.isalpha()]
        
        for _ in range(num_words):
            word_len = random.randint(
                text_config['min_word_length'], 
                text_config['max_word_length']
            )
            
            # Different numeric patterns
            pattern_choice = random.random()
            
            if pattern_choice < 0.4:  # 40% pure numbers
                # Pure numeric sequences (years, codes, etc.)
                word = ''.join(random.choices(digits, k=word_len))
            elif pattern_choice < 0.7:  # 30% alphanumeric (like ROOM101, R2D2)
                if word_len >= 3:
                    # Letter prefix + number suffix
                    letter_len = random.randint(1, word_len - 2)
                    number_len = word_len - letter_len
                    letter_part = ''.join(random.choices(letters, k=letter_len))
                    number_part = ''.join(random.choices(digits, k=number_len))
                    word = letter_part + number_part
                else:
                    word = ''.join(random.choices(digits, k=word_len))
            elif pattern_choice < 0.85:  # 15% number prefix + letters
                if word_len >= 3:
                    # Number prefix + letter suffix  
                    number_len = random.randint(1, word_len - 2)
                    letter_len = word_len - number_len
                    number_part = ''.join(random.choices(digits, k=number_len))
                    letter_part = ''.join(random.choices(letters, k=letter_len))
                    word = number_part + letter_part
                else:
                    word = ''.join(random.choices(digits, k=word_len))
            else:  # 15% mixed with contractions
                if "'" in self.charset and word_len >= 4:
                    # Create realistic patterns like "DON'T" but with numbers "20'S", "90'S"
                    if random.random() < 0.3:  # Number + 'S pattern
                        number_part = ''.join(random.choices(digits, k=word_len - 2))
                        word = number_part + "'S"
                    else:  # Regular contraction-like
                        part1_len = random.randint(1, word_len - 3)
                        part2_len = word_len - part1_len - 1
                        part1 = ''.join(random.choices(letters, k=part1_len))
                        part2 = ''.join(random.choices(letters, k=part2_len))
                        word = f"{part1}'{part2}"
                else:
                    word = ''.join(random.choices(digits, k=word_len))
            
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
                if self.word_list and random.random() < (1 - self.random_text_ratio):
                    # Use vocabulary
                    word = random.choice(self.word_list)
                else:
                    # Random Aurebesh characters
                    word_len = random.randint(
                        text_config['min_word_length'], 
                        text_config['max_word_length']
                    )
                    available_chars = list(self.charset)
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
    
    def _check_word_overlap(self, new_bbox: List[int], existing_annotations: List[dict], min_gap: int = 5) -> bool:
        """Check if a word bounding box overlaps with existing word annotations.
        
        Args:
            new_bbox: [x1, y1, x2, y2] of the new word
            existing_annotations: List of existing word annotations with 'bbox' field
            min_gap: Minimum gap in pixels between words
            
        Returns:
            True if there is an overlap, False otherwise
        """
        for existing in existing_annotations:
            exist_bbox = existing['bbox']
            # Check if boxes overlap with min_gap spacing
            if (new_bbox[0] - min_gap < exist_bbox[2] and 
                new_bbox[2] + min_gap > exist_bbox[0] and
                new_bbox[1] - min_gap < exist_bbox[3] and 
                new_bbox[3] + min_gap > exist_bbox[1]):
                return True
        return False
    
    def _get_background(self, size: Tuple[int, int]) -> Image.Image:
        """Get background image or generate synthetic one."""
        if random.random() < self.config['style']['color']['synthetic_bg_prob']:
            # Generate synthetic background based on pattern probabilities
            patterns_config = self.config['style']['color']['synthetic_patterns']
            pattern_names = list(patterns_config.keys())
            pattern_probs = [patterns_config[name] for name in pattern_names]
            
            # Choose pattern based on probabilities
            pattern = random.choices(pattern_names, weights=pattern_probs)[0]
            
            if pattern == 'solid_color':
                return self._generate_solid_color_bg(size)
            elif pattern == 'gradient':
                return self._generate_gradient_bg(size)
            elif pattern == 'random_lines':
                return self._generate_random_lines_bg(size)
            elif pattern == 'geometric_shapes':
                return self._generate_geometric_shapes_bg(size)
            elif pattern == 'noise_texture':
                return self._generate_noise_texture_bg(size)
            elif pattern == 'pseudo_characters':
                return self._generate_pseudo_characters_bg(size)
            else:
                # Fallback to solid color
                return self._generate_solid_color_bg(size)
        else:
            # Use real background
            bg_path = random.choice(self.backgrounds)
            bg = Image.open(bg_path).convert('RGB')
            bg = bg.resize(size, Image.Resampling.LANCZOS)
            return bg
    
    def _generate_solid_color_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate solid color background."""
        bg = Image.new('RGB', size)
        draw = ImageDraw.Draw(bg)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([0, 0, size[0], size[1]], fill=color)
        return bg
    
    def _generate_gradient_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate gradient background."""
        bg = Image.new('RGB', size)
        draw = ImageDraw.Draw(bg)
        
        # Choose gradient direction
        if random.random() < 0.5:
            # Vertical gradient
            for y in range(size[1]):
                color = tuple(
                    int(128 + 127 * np.sin(y / size[1] * np.pi + random.random() * 2 * np.pi))
                    for _ in range(3)
                )
                draw.line([(0, y), (size[0], y)], fill=color)
        else:
            # Horizontal gradient
            for x in range(size[0]):
                color = tuple(
                    int(128 + 127 * np.sin(x / size[0] * np.pi + random.random() * 2 * np.pi))
                    for _ in range(3)
                )
                draw.line([(x, 0), (x, size[1])], fill=color)
        
        return bg
    
    def _generate_random_lines_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate background with random lines and curves that resemble character strokes."""
        bg = Image.new('RGB', size)
        draw = ImageDraw.Draw(bg)
        
        # Base background color
        base_color = tuple(random.randint(180, 255) for _ in range(3))
        draw.rectangle([0, 0, size[0], size[1]], fill=base_color)
        
        # Add random lines
        num_lines = random.randint(20, 40)
        for _ in range(num_lines):
            # Line color - similar contrast to text
            line_color = tuple(random.randint(0, 150) for _ in range(3))
            
            # Line width similar to character strokes
            line_width = random.randint(2, 5)
            
            # Choose line type
            if random.random() < 0.5:
                # Straight line
                x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
                x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
            else:
                # Curved line (using bezier curve approximation)
                points = []
                num_points = random.randint(3, 6)
                for _ in range(num_points):
                    x = random.randint(0, size[0])
                    y = random.randint(0, size[1])
                    points.append((x, y))
                
                # Draw curve as connected line segments
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=line_color, width=line_width)
        
        return bg
    
    def _generate_geometric_shapes_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate background with geometric shapes that could be mistaken for characters."""
        bg = Image.new('RGB', size)
        draw = ImageDraw.Draw(bg)
        
        # Base background color
        base_color = tuple(random.randint(200, 255) for _ in range(3))
        draw.rectangle([0, 0, size[0], size[1]], fill=base_color)
        
        # Add geometric shapes
        num_shapes = random.randint(15, 30)
        for _ in range(num_shapes):
            shape_color = tuple(random.randint(0, 180) for _ in range(3))
            opacity = random.randint(100, 200)  # Semi-transparent
            
            # Character-sized shapes
            shape_size = random.randint(20, 60)
            x = random.randint(0, size[0] - shape_size)
            y = random.randint(0, size[1] - shape_size)
            
            # Create overlay for transparency
            overlay = Image.new('RGBA', size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])
            
            if shape_type == 'rectangle':
                overlay_draw.rectangle(
                    [x, y, x + shape_size, y + shape_size],
                    fill=shape_color + (opacity,)
                )
            elif shape_type == 'ellipse':
                overlay_draw.ellipse(
                    [x, y, x + shape_size, y + shape_size],
                    fill=shape_color + (opacity,)
                )
            else:  # triangle
                points = [
                    (x + shape_size // 2, y),
                    (x, y + shape_size),
                    (x + shape_size, y + shape_size)
                ]
                overlay_draw.polygon(points, fill=shape_color + (opacity,))
            
            # Composite overlay onto background
            bg = Image.alpha_composite(bg.convert('RGBA'), overlay).convert('RGB')
        
        return bg
    
    def _generate_noise_texture_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate background with noise textures that have high-frequency components."""
        bg = Image.new('RGB', size)
        
        # Base color
        base_color = random.randint(180, 255)
        bg_array = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
        
        # Choose noise type
        noise_type = random.choice(['perlin', 'salt_pepper', 'grid'])
        
        if noise_type == 'perlin':
            # Simple Perlin-like noise using sine waves
            freq_x = random.uniform(0.01, 0.05)
            freq_y = random.uniform(0.01, 0.05)
            
            for y in range(size[1]):
                for x in range(size[0]):
                    noise_val = (
                        np.sin(x * freq_x) * np.sin(y * freq_y) +
                        np.sin(x * freq_x * 2.1) * np.sin(y * freq_y * 2.1) * 0.5
                    )
                    intensity = int(base_color + noise_val * 50)
                    intensity = max(0, min(255, intensity))
                    bg_array[y, x] = [intensity, intensity, intensity]
        
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            noise_density = 0.05
            for y in range(size[1]):
                for x in range(size[0]):
                    if random.random() < noise_density:
                        # Random black or white pixel
                        value = 0 if random.random() < 0.5 else 255
                        bg_array[y, x] = [value, value, value]
        
        else:  # grid
            # Grid pattern
            grid_size = random.randint(20, 40)
            line_color = random.randint(0, 150)
            
            # Vertical lines
            for x in range(0, size[0], grid_size):
                bg_array[:, x:x+2] = line_color
            
            # Horizontal lines
            for y in range(0, size[1], grid_size):
                bg_array[y:y+2, :] = line_color
        
        return Image.fromarray(bg_array)
    
    def _generate_pseudo_characters_bg(self, size: Tuple[int, int]) -> Image.Image:
        """Generate background with pseudo-character symbols."""
        bg = Image.new('RGB', size)
        draw = ImageDraw.Draw(bg)
        
        # Base background color
        base_color = tuple(random.randint(200, 255) for _ in range(3))
        draw.rectangle([0, 0, size[0], size[1]], fill=base_color)
        
        # Pseudo-character patterns
        patterns = [
            # Cross
            lambda x, y, s: [((x, y + s//3), (x + s, y + s//3)), 
                           ((x + s//2, y), (x + s//2, y + s))],
            # L shape
            lambda x, y, s: [((x, y), (x, y + s)), 
                           ((x, y + s), (x + s//2, y + s))],
            # T shape
            lambda x, y, s: [((x, y), (x + s, y)), 
                           ((x + s//2, y), (x + s//2, y + s))],
            # Angle bracket
            lambda x, y, s: [((x, y), (x + s//2, y + s//2)), 
                           ((x + s//2, y + s//2), (x, y + s))],
            # Square bracket
            lambda x, y, s: [((x, y), (x, y + s)), 
                           ((x, y), (x + s//3, y)), 
                           ((x, y + s), (x + s//3, y + s))],
        ]
        
        # Add pseudo-characters
        num_symbols = random.randint(10, 25)
        for _ in range(num_symbols):
            symbol_color = tuple(random.randint(0, 150) for _ in range(3))
            symbol_size = random.randint(20, 50)
            line_width = random.randint(2, 5)
            
            # Random position
            x = random.randint(0, size[0] - symbol_size)
            y = random.randint(0, size[1] - symbol_size)
            
            # Random rotation angle
            angle = random.uniform(0, 360)
            
            # Choose pattern
            pattern = random.choice(patterns)
            lines = pattern(0, 0, symbol_size)
            
            # Draw pattern with rotation
            for line in lines:
                # Rotate line points around center
                cx, cy = symbol_size // 2, symbol_size // 2
                rotated_line = []
                
                for px, py in line:
                    # Translate to origin
                    px -= cx
                    py -= cy
                    
                    # Rotate
                    angle_rad = np.radians(angle)
                    new_x = px * np.cos(angle_rad) - py * np.sin(angle_rad)
                    new_y = px * np.sin(angle_rad) + py * np.cos(angle_rad)
                    
                    # Translate back and offset to position
                    new_x += cx + x
                    new_y += cy + y
                    
                    rotated_line.append((new_x, new_y))
                
                draw.line(rotated_line, fill=symbol_color, width=line_width)
        
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
        """Render multiple lines of text with word-level bboxes."""
        annotations = []
        current_y = y
        effects = self.config['style']['effects']
        
        for i, (line_text, font_size) in enumerate(zip(lines, font_sizes)):
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except:
                font = ImageFont.load_default()
            
            # Get line metrics for positioning
            line_bbox = draw.textbbox((0, 0), line_text, font=font)
            text_height = line_bbox[3] - line_bbox[1]
            actual_y_offset = -line_bbox[1]
            
            # Calculate base line position
            line_y = current_y + actual_y_offset
            
            # Split line into words and render each word separately
            words = line_text.split()
            current_x = x
            
            for word in words:
                # Get word metrics
                word_bbox = draw.textbbox((0, 0), word, font=font)
                actual_x_offset = -word_bbox[0]
                
                # Adjust word position
                word_x = current_x + actual_x_offset
                word_y = line_y
                
                # Apply effects (shadow/border) for each word
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
                
                # Get precise bbox for this word
                final_word_bbox = draw.textbbox((word_x, word_y), word, font=font)
                
                # Create annotation for this word
                annotation = {
                    'text': word,
                    'bbox': [final_word_bbox[0], final_word_bbox[1], 
                            final_word_bbox[2], final_word_bbox[3]],
                    'polygon': [
                        [final_word_bbox[0], final_word_bbox[1]],
                        [final_word_bbox[2], final_word_bbox[1]],
                        [final_word_bbox[2], final_word_bbox[3]],
                        [final_word_bbox[0], final_word_bbox[3]]
                    ]
                }
                annotations.append(annotation)
                
                # Move to next word position (add word width + space)
                word_width = word_bbox[2] - word_bbox[0]
                space_width = draw.textbbox((0, 0), ' ', font=font)[2]
                current_x += word_width + space_width
            
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
            min_spacing = max(25, int(base_min_spacing * spacing_factor))  # Increased minimum spacing
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
                if not words:
                    # Empty text - create empty lines
                    lines = ["", ""]
                    font_sizes = [font_size, font_size]
                else:
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
                    
                    # Ensure we have at least one line
                    if not lines:
                        lines = [block_text if block_text else ""]
                    
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
                elif len(words) == 1:
                    # Single word - duplicate it
                    lines = [words[0], words[0]]
                    font_sizes = [font_size, max(15, font_size // 2)]
                else:
                    # Empty text - use empty lines
                    lines = ["", ""]
                    font_sizes = [font_size, max(15, font_size // 2)]
            else:
                # Normal single line
                lines = [block_text]
                font_sizes = [font_size]
            
            # Skip text blocks with only empty lines
            if all(not line.strip() for line in lines):
                continue
            
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
            x, y = 0, 0  # Initialize variables
            placement_width, placement_height = text_width, text_height  # Initialize placement dimensions
            
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
                
                if not overlap:
                    # Found non-overlapping position
                    occupied_regions.append(new_region)
                    
                    # Mark spatial grid as occupied
                    self._mark_grid_occupied(spatial_grid, new_region, grid_cell_size)
                    break
            else:
                # Could not find non-overlapping position after max_attempts
                self.logger.debug(f"Skipping text block {block_idx} due to no available space")
                continue  # Skip this text block
            
            # Draw text with pre-determined rotation
            if will_rotate:
                # For rotated text, we'll render each word separately and then combine
                words = block_text.split()
                word_annotations = []
                
                for word in words:
                            # Get word dimensions
                            word_bbox = draw.textbbox((0, 0), word, font=font)
                            word_width = word_bbox[2] - word_bbox[0]
                            word_height = word_bbox[3] - word_bbox[1]
                            
                            # Create text image for this word with precise dimensions
                            padding = 20
                            temp_size = (int(word_width) + 2*padding, int(word_height) + 2*padding)
                            text_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
                            text_draw = ImageDraw.Draw(text_img)
                            
                            # Center word in temporary image
                            temp_x = padding
                            temp_y = padding
                            
                            # Draw word with effects on temporary image
                            # Shadow
                            if random.random() < effects['shadow_prob']:
                                shadow_offset = random.randint(2, 5)
                                shadow_color = tuple(int(c * 0.3) for c in text_color) + (255,)
                                text_draw.text((temp_x + shadow_offset, temp_y + shadow_offset), 
                                             word, font=font, fill=shadow_color)
                            
                            # Border
                            if random.random() < effects['border_prob']:
                                for dx in [-1, 0, 1]:
                                    for dy in [-1, 0, 1]:
                                        if dx != 0 or dy != 0:
                                            text_draw.text((temp_x + dx, temp_y + dy), 
                                                         word, font=font, fill=(0, 0, 0, 255))
                            
                            # Main text
                            text_draw.text((temp_x, temp_y), word, font=font, fill=text_color + (255,))
                            
                            # Rotate the word image
                            rotated_word = text_img.rotate(-angle, expand=True, fillcolor=(0, 0, 0, 0))
                            
                            # Calculate paste position for this word
                            rot_w, rot_h = rotated_word.size
                            
                            # Try to find a non-overlapping position for this word
                            max_word_attempts = 50
                            word_placed = False
                            
                            for word_attempt in range(max_word_attempts):
                                if word_attempt == 0:
                                    # First attempt: use center position
                                    paste_x = x + int(placement_width - rot_w) // 2
                                    paste_y = y + int(placement_height - rot_h) // 2
                                else:
                                    # Subsequent attempts: random position within the block area
                                    paste_x = x + random.randint(0, max(0, placement_width - rot_w))
                                    paste_y = y + random.randint(0, max(0, placement_height - rot_h))
                                
                                # Final boundary check
                                if (paste_x < 0 or paste_y < 0 or 
                                    paste_x + rot_w > image_size[0] or 
                                    paste_y + rot_h > image_size[1]):
                                    continue  # Try another position
                                
                                # Calculate the actual bounding box for overlap check
                                # First, find the actual bounds by analyzing the rotated image
                                rot_array = np.array(rotated_word)
                                alpha_channel = rot_array[:, :, 3]
                                y_indices, x_indices = np.where(alpha_channel > 0)
                                
                                if len(x_indices) > 0 and len(y_indices) > 0:
                                    # Calculate actual word bounds
                                    actual_x1 = paste_x + np.min(x_indices)
                                    actual_y1 = paste_y + np.min(y_indices)
                                    actual_x2 = paste_x + np.max(x_indices) + 1
                                    actual_y2 = paste_y + np.max(y_indices) + 1
                                    
                                    # Check for overlap with existing words
                                    temp_bbox = [actual_x1, actual_y1, actual_x2, actual_y2]
                                    # Check overlap with both global annotations and current block's words
                                    combined_annotations = annotations + word_annotations
                                    if not self._check_word_overlap(temp_bbox, combined_annotations):
                                        # No overlap, we can place the word here
                                        word_placed = True
                                        break
                            
                            if not word_placed:
                                self.logger.debug(f"Could not find non-overlapping position for word: {word}")
                                continue  # Skip this word
                            
                            # Paste rotated word onto background
                            bg.paste(rotated_word, (paste_x, paste_y), rotated_word)
                            
                            # Calculate rotated polygon for this word
                            # The key insight: use the actual bounding box coordinates that were correctly calculated
                            # and transform them back to get the polygon corners
                            
                            # actual_x1, actual_y1, actual_x2, actual_y2 are the correct bounding box coordinates
                            # We need to find the original corners that would produce this bounding box after rotation
                            
                            # Get the actual text boundaries in the temp image (before rotation)
                            temp_text_bbox = text_draw.textbbox((temp_x, temp_y), word, font=font)
                            
                            # Calculate the actual text rectangle corners in temp image coordinates
                            text_left = temp_text_bbox[0]
                            text_top = temp_text_bbox[1] 
                            text_right = temp_text_bbox[2]
                            text_bottom = temp_text_bbox[3]
                            
                            # Transform these coordinates to be relative to the temp image center
                            temp_center_x = temp_size[0] / 2
                            temp_center_y = temp_size[1] / 2
                            
                            corners = [
                                (text_left - temp_center_x, text_top - temp_center_y),
                                (text_right - temp_center_x, text_top - temp_center_y),
                                (text_right - temp_center_x, text_bottom - temp_center_y),
                                (text_left - temp_center_x, text_bottom - temp_center_y)
                            ]
                            
                            # Rotate corners
                            angle_rad = np.radians(angle)
                            cos_a = np.cos(angle_rad)
                            sin_a = np.sin(angle_rad)
                            
                            rotated_corners = []
                            for cx, cy in corners:
                                rx = cx * cos_a - cy * sin_a
                                ry = cx * sin_a + cy * cos_a
                                
                                # Transform to final image coordinates
                                final_x = paste_x + rot_w / 2 + rx
                                final_y = paste_y + rot_h / 2 + ry
                                
                                rotated_corners.append([int(final_x), int(final_y)])
                            
                            # Create annotation for this word
                            word_annotation = {
                                'text': word,
                                'bbox': [actual_x1, actual_y1, actual_x2, actual_y2],
                                'polygon': rotated_corners
                            }
                            word_annotations.append(word_annotation)
                
                # Add all word annotations
                annotations.extend(word_annotations)
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
                    # Single line text - render word by word
                    words = block_text.split()
                    current_x = x
                    
                    for word in words:
                        # Get word metrics
                        word_bbox = draw.textbbox((0, 0), word, font=font)
                        actual_x_offset = -word_bbox[0]  # Left bearing
                        actual_y_offset = -word_bbox[1]  # Top bearing
                        
                        # Adjust position to account for font metrics
                        word_x = current_x + actual_x_offset
                        word_y = y + actual_y_offset
                        
                        # Final boundary check - ensure word won't exceed image bounds
                        final_bbox_check = draw.textbbox((word_x, word_y), word, font=font)
                        if (final_bbox_check[0] < 0 or final_bbox_check[1] < 0 or
                            final_bbox_check[2] > image_size[0] or final_bbox_check[3] > image_size[1]):
                            break  # Skip remaining words in this line
                        
                        # Shadow
                        if random.random() < effects['shadow_prob']:
                            shadow_offset = random.randint(2, 5)
                            shadow_color = tuple(int(c * 0.3) for c in text_color)
                            draw.text((word_x + shadow_offset, word_y + shadow_offset), 
                                    word, font=font, fill=shadow_color)
                        
                        # Border
                        if random.random() < effects['border_prob']:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        draw.text((word_x + dx, word_y + dy), 
                                                word, font=font, fill=(0, 0, 0))
                        
                        # Main text
                        draw.text((word_x, word_y), word, font=font, fill=text_color)
                        
                        # Create precise bounding box using actual text metrics
                        final_bbox = draw.textbbox((word_x, word_y), word, font=font)
                        
                        # Create annotation with precise coordinates
                        annotation = {
                            'text': word,
                            'bbox': [final_bbox[0], final_bbox[1], final_bbox[2], final_bbox[3]],
                            'polygon': [
                                [final_bbox[0], final_bbox[1]],
                                [final_bbox[2], final_bbox[1]],
                                [final_bbox[2], final_bbox[3]],
                                [final_bbox[0], final_bbox[3]]
                            ]
                        }
                        
                        annotations.append(annotation)
                        
                        # Move to next word position (add word width + space)
                        word_width = word_bbox[2] - word_bbox[0]
                        space_width = draw.textbbox((0, 0), ' ', font=font)[2]
                        current_x += word_width + space_width
                    break
        
        # Log placement statistics
        placed_blocks = len(annotations)
        self.logger.debug(f"Successfully placed {placed_blocks}/{num_text_blocks} text blocks with min_spacing={min_spacing}")
        
        return bg, annotations
    
    def _save_debug_image(self, image: Image.Image, annotations: List[Dict], output_path: Path):
        """Save image with bounding boxes drawn for debugging."""
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        
        for ann in annotations:
            bbox = ann['bbox']
            polygon = ann['polygon']
            
            # Draw bounding box in red (thinner line)
            draw.rectangle(bbox, outline='red', width=1)
            
            # Draw polygon in blue (thicker line to emphasize the actual text region)
            if len(polygon) >= 3:
                flat_polygon = [coord for point in polygon for coord in point]
                draw.polygon(flat_polygon, outline='blue', width=3)
                
                # Also draw polygon vertices as small circles
                for point in polygon:
                    x, y = point
                    draw.ellipse([x-3, y-3, x+3, y+3], fill='green', outline='green')
            
            # Draw text label with background for better visibility
            text_to_show = ann['text'][:20]
            try:
                # Try to get text size for background
                font = ImageFont.load_default()
                bbox_text = draw.textbbox((bbox[0], bbox[1] - 20), text_to_show, font=font)
                # Draw background rectangle
                draw.rectangle([bbox_text[0]-2, bbox_text[1]-2, bbox_text[2]+2, bbox_text[3]+2], fill='black')
            except:
                pass
            
            text_pos = (bbox[0], bbox[1] - 20)
            draw.text(text_pos, text_to_show, fill='yellow')
        
        debug_img.save(output_path)

    def _perspective_crop_polygon(self, image: Image.Image, polygon: List[List[int]]) -> Image.Image:
        """Apply perspective transform to crop rotated polygon as upright rectangle."""
        # Convert polygon to numpy array
        polygon_np = np.array(polygon, dtype=np.float32)
        
        # Calculate the width and height of the output rectangle
        # Use the distances between opposite corners to determine dimensions
        width1 = np.linalg.norm(polygon_np[1] - polygon_np[0])
        width2 = np.linalg.norm(polygon_np[2] - polygon_np[3])
        height1 = np.linalg.norm(polygon_np[3] - polygon_np[0])
        height2 = np.linalg.norm(polygon_np[2] - polygon_np[1])
        
        # Use maximum width and height to avoid cutting off text
        max_width = int(max(width1, width2))
        max_height = int(max(height1, height2))
        
        # Ensure minimum size
        max_width = max(max_width, 20)
        max_height = max(max_height, 20)
        
        # Define destination rectangle (upright)
        dst_points = np.array([
            [0, 0],
            [max_width, 0],
            [max_width, max_height],
            [0, max_height]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(polygon_np, dst_points)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image_np, transform_matrix, (max_width, max_height))
        
        # Convert back to PIL image
        return Image.fromarray(warped)

    def _save_cropped_image(self, image: Image.Image, text: str, cropped_images_dir: Path, image_name: str, crop_idx: int) -> str:
        """Save cropped text image for recognizer."""
        # Create filename for cropped image
        crop_filename = f"{image_name.stem}_{crop_idx:02d}.jpg"
        crop_path = cropped_images_dir / crop_filename
        
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            rgb_image.save(crop_path, 'JPEG', quality=95)
        else:
            image.save(crop_path, 'JPEG', quality=95)
        
        return crop_filename
    
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
        global_image_counter = 1  # Global counter for image naming
        
        for split_name, split_size in splits.items():
            self.logger.info(f"Generating {split_name} split with {split_size} images")
            
            # Setup directories
            split_dir = ensure_dir(self.output_dir / split_name)
            images_dir = ensure_dir(split_dir / 'images')
            
            # Setup cropped images directory for recognizer
            cropped_dir = ensure_dir(split_dir / 'cropped')
            cropped_images_dir = ensure_dir(cropped_dir / 'images')
            
            # Annotations in new format
            annotations = {}
            # Cropped images annotations for recognizer
            cropped_annotations = {}
            
            generated_count = 0
            attempts = 0
            max_attempts = split_size * 2  # Allow up to 2x attempts to get required images
            
            with tqdm(total=split_size, desc=f"Generating {split_name}") as pbar:
                while generated_count < split_size and attempts < max_attempts:
                    # Generate text
                    text = self._generate_text()
                    
                    # Handle empty text case
                    if not text.strip():
                        # Generate background-only image for empty text
                        bg = self._get_background((self.resolution, self.resolution))
                        
                        # Convert to RGB if needed
                        if bg.mode == 'RGBA':
                            rgb_image = Image.new('RGB', bg.size, (255, 255, 255))
                            rgb_image.paste(bg, mask=bg.split()[3])
                            image_aug = rgb_image
                        else:
                            image_aug = bg
                        
                        # Save detection image
                        image_name = f"img_{global_image_counter:04d}.jpg"
                        image_path = images_dir / image_name
                        image_aug.save(image_path, 'JPEG', quality=95)
                        
                        # Save debug image (if debug mode enabled)
                        if self.debug and generated_count < 20:
                            debug_dir = ensure_dir(split_dir / 'debug')
                            debug_name = f"img_{global_image_counter:04d}_debug.png"
                            debug_path = debug_dir / debug_name
                            self._save_debug_image(image_aug, [], debug_path)
                        
                        # Add empty annotation
                        annotations[image_name] = {
                            'polygons': [],
                            'texts': []
                        }
                        
                        generated_count += 1
                        global_idx += 1
                        global_image_counter += 1
                        attempts += 1
                        pbar.update(1)
                        continue
                    
                    # Sample font
                    font_path = self._sample_font()
                    
                    # Render text on image
                    image, text_annotations = self._render_text_on_image(
                        text, font_path, (self.resolution, self.resolution)
                    )
                    
                    attempts += 1
                    
                    # Handle images with no text annotations (e.g., failed rendering)
                    if not text_annotations:
                        self.logger.debug(f"No text placed on image attempt {attempts}, saving as background-only image")
                        
                        # Convert to RGB if needed
                        if image.mode == 'RGBA':
                            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                            rgb_image.paste(image, mask=image.split()[3])
                            image_aug = rgb_image
                        else:
                            image_aug = image
                        
                        # Save detection image
                        image_name = f"img_{global_image_counter:04d}.jpg"
                        image_path = images_dir / image_name
                        image_aug.save(image_path, 'JPEG', quality=95)
                        
                        # Save debug image (if debug mode enabled)
                        if self.debug and generated_count < 20:
                            debug_dir = ensure_dir(split_dir / 'debug')
                            debug_name = f"img_{global_image_counter:04d}_debug.png"
                            debug_path = debug_dir / debug_name
                            self._save_debug_image(image_aug, [], debug_path)
                        
                        # Add empty annotation
                        annotations[image_name] = {
                            'polygons': [],
                            'texts': []
                        }
                        
                        generated_count += 1
                        global_idx += 1
                        global_image_counter += 1
                        pbar.update(1)
                        continue
                    
                    # Apply augmentations with bbox transformation
                    image_np = np.array(image)
                    
                    # For augmentations, we'll use keypoints to track polygon corners
                    # This ensures rotated polygons are properly transformed
                    bboxes = []
                    labels = []
                    keypoints = []
                    valid_annotations = []
                    
                    for ann in text_annotations:
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
                        
                        # Store polygon corners as keypoints for transformation
                        polygon_kps = []
                        for point in ann['polygon']:
                            # Each keypoint is [x, y] - we'll add visibility flag 1
                            polygon_kps.extend([point[0], point[1], 1])
                        keypoints.append(polygon_kps)
                        
                        valid_annotations.append(ann)
                    
                    # Skip if no valid bboxes remain
                    if not bboxes:
                        self.logger.warning("No valid bboxes after filtering, retrying...")
                        continue
                    
                    # Setup keypoint params for albumentations
                    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)
                    
                    # Create augmentation pipeline with keypoint support
                    aug_with_keypoints = A.Compose(
                        self.augmentations.transforms,
                        bbox_params=A.BboxParams(
                            format='pascal_voc',
                            label_fields=['labels'],
                            min_visibility=0.3
                        ),
                        keypoint_params=keypoint_params
                    )
                    
                    # Flatten keypoints for albumentations format
                    flat_keypoints = []
                    for kps in keypoints:
                        # Convert from [x1,y1,v1, x2,y2,v2, ...] to [(x1,y1), (x2,y2), ...]
                        for i in range(0, len(kps), 3):
                            flat_keypoints.append((kps[i], kps[i+1]))
                    
                    # Apply augmentations
                    augmented = aug_with_keypoints(
                        image=image_np,
                        bboxes=bboxes,
                        labels=labels,
                        keypoints=flat_keypoints
                    )
                    
                    image_aug = Image.fromarray(augmented['image'])
                    augmented_bboxes = augmented['bboxes']
                    augmented_labels = augmented['labels']
                    augmented_keypoints = augmented['keypoints']
                    
                    # Reconstruct polygons from augmented keypoints
                    keypoint_idx = 0
                    augmented_polygons = []
                    for _ in range(len(valid_annotations)):
                        # Each annotation has 4 keypoints (polygon corners)
                        polygon = []
                        for _ in range(4):
                            if keypoint_idx < len(augmented_keypoints):
                                kp = augmented_keypoints[keypoint_idx]
                                polygon.append([int(kp[0]), int(kp[1])])
                                keypoint_idx += 1
                        augmented_polygons.append(polygon)
                    
                    # Update text_annotations with transformed bboxes and polygons
                    updated_annotations = []
                    for bbox, label, polygon in zip(augmented_bboxes, augmented_labels, augmented_polygons):
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
                        
                        # Use the transformed polygon
                        updated_ann['polygon'] = polygon
                        
                        updated_annotations.append(updated_ann)
                    
                    # Use updated annotations for saving
                    text_annotations = updated_annotations
                    
                    # Save detection image
                    image_name = f"img_{global_image_counter:04d}.jpg"
                    image_path = images_dir / image_name
                    # Convert to RGB before saving as JPEG
                    if image_aug.mode == 'RGBA':
                        rgb_image = Image.new('RGB', image_aug.size, (255, 255, 255))
                        rgb_image.paste(image_aug, mask=image_aug.split()[3])
                        rgb_image.save(image_path, 'JPEG', quality=95)
                    else:
                        image_aug.save(image_path, 'JPEG', quality=95)
                    
                    # Save debug image with bboxes (if debug mode enabled)
                    if self.debug and generated_count < 20:  # Save debug for first 20 images
                        debug_dir = ensure_dir(split_dir / 'debug')
                        debug_name = f"img_{global_image_counter:04d}_debug.png"
                        debug_path = debug_dir / debug_name
                        self._save_debug_image(image_aug, text_annotations, debug_path)
                    
                    # Prepare data for new labels format
                    polygons = []
                    texts = []
                    crop_idx = 1  # Counter for cropped images per main image
                    
                    # Process each text instance
                    for ann in text_annotations:
                        # Get polygon for perspective transform
                        polygon = ann['polygon']
                        bbox = ann['bbox']
                        
                        # Skip if polygon is outside image boundaries
                        # Check all polygon points
                        polygon_valid = True
                        for point in polygon:
                            if (point[0] < 0 or point[1] < 0 or 
                                point[0] > self.resolution or point[1] > self.resolution):
                                polygon_valid = False
                                break
                        
                        if not polygon_valid:
                            self.logger.debug(f"Skipping out-of-bounds polygon during crop: {ann['text']}")
                            continue
                        
                        # Apply perspective transformation to crop rotated text
                        try:
                            text_crop = self._perspective_crop_polygon(image_aug, polygon)
                        except Exception as e:
                            self.logger.debug(f"Failed to crop polygon for text '{ann['text']}': {e}")
                            continue
                        
                        # Skip very small crops
                        if text_crop.size[0] < 10 or text_crop.size[1] < 10:
                            self.logger.debug(f"Skipping too small crop: {ann['text']}")
                            continue
                        
                        # Save cropped image for recognizer
                        crop_filename = self._save_cropped_image(
                            text_crop, ann['text'], cropped_images_dir, 
                            Path(image_name), crop_idx
                        )
                        cropped_annotations[crop_filename] = ann['text']
                        crop_idx += 1
                        
                        # Add polygon and text to arrays
                        polygons.append(ann['polygon'])
                        texts.append(ann['text'])
                    
                    # Add annotation (either with content or empty)
                    annotations[image_name] = {
                        'polygons': polygons,
                        'texts': texts
                    }
                    
                    generated_count += 1
                    global_idx += 1
                    global_image_counter += 1
                    pbar.update(1)
                
                if generated_count < split_size:
                    self.logger.warning(f"Could only generate {generated_count}/{split_size} images after {attempts} attempts")
            
            # Save annotations as labels.json
            labels_path = split_dir / 'labels.json'
            with open(labels_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            # Save cropped annotations for recognizer
            cropped_labels_path = cropped_dir / 'labels.json'
            with open(cropped_labels_path, 'w') as f:
                json.dump(cropped_annotations, f, indent=2)
            
            self.logger.info(f"Completed {split_name} split with {len(cropped_annotations)} cropped images")
        
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