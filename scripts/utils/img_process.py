"""
Image processing utilities for Aurebesh OCR.
Contains functions for cropping and transforming detected text regions.
"""

from typing import List
import numpy as np
from PIL import Image
import cv2


# MARGIN_RATIO = 0.12  # # 端欠け防止のマージン係数（高さH比）
MARGIN_RATIO = 0

def perspective_crop_polygon(image: Image.Image, polygon: List[List[int]]) -> Image.Image:
    """
    Apply perspective transform to crop rotated polygon as upright rectangle.
    
    This function performs the same cropping and transformation used during
    dataset generation to ensure consistency between training and inference.
    
    Args:
        image: Input PIL image (should be in RGB format)
        polygon: List of 4 corner points as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                The polygon should represent the corners of the text region.
    
    Returns:
        PIL Image: Cropped and perspective-corrected text region
        
    Raises:
        ValueError: If polygon doesn't have exactly 4 points
        cv2.error: If perspective transformation fails
    """
    if len(polygon) != 4:
        raise ValueError(f"Polygon must have exactly 4 points, got {len(polygon)}")
    
    # Convert polygon to numpy array
    polygon_np = np.array(polygon, dtype=np.float32)
    
    # Calculate polygon height to determine margin amounts
    height1 = np.linalg.norm(polygon_np[3] - polygon_np[0])
    height2 = np.linalg.norm(polygon_np[2] - polygon_np[1])
    H = max(height1, height2)
    
    # Calculate margin amounts based on polygon height
    margin_pixels = MARGIN_RATIO * H  # Use ratio as base margin
    
    # Simple expansion: move each point outward from polygon center
    # This is safer and less likely to cause rotation issues
    center = np.mean(polygon_np, axis=0)
    expanded_polygon_np = center + (polygon_np - center) * (1 + margin_pixels / H)
    
    # Ensure expanded polygon stays within image boundaries
    image_width, image_height = image.size
    expanded_polygon_np[:, 0] = np.clip(expanded_polygon_np[:, 0], 0, image_width - 1)
    expanded_polygon_np[:, 1] = np.clip(expanded_polygon_np[:, 1], 0, image_height - 1)
    
    # Use expanded polygon for transformation
    polygon_np = expanded_polygon_np
    
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
