"""Image processing utilities for Aurebesh OCR."""

from typing import List
import numpy as np
from PIL import Image
import cv2


def perspective_crop_polygon(image: Image.Image, polygon: List[List[int]]) -> Image.Image:
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
