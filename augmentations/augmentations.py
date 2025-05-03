# augmentations/augmentations.py

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random

# Define your augmentation techniques here.
# Each function should accept a PIL Image and parameters, and return a new PIL Image.

def rotate(image: Image.Image, degrees: float = 0, expand: bool = False) -> Image.Image:
    """Rotates the image."""
    return image.rotate(degrees, expand=expand)

def flip_horizontal(image: Image.Image) -> Image.Image:
    """Flips the image horizontally."""
    return ImageOps.mirror(image)

def flip_vertical(image: Image.Image) -> Image.Image:
    """Flips the image vertically."""
    return ImageOps.flip(image)

def color_jitter(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0, color: float = 1.0, sharpness: float = 1.0) -> Image.Image:
    """Adjusts brightness, contrast, color, and sharpness."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

def gaussian_blur(image: Image.Image, radius: float = 0) -> Image.Image:
    """Applies Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius))

# Add more augmentations as needed...

# --- Information for the frontend ---
# This dictionary maps technique names (strings) to their corresponding functions
AUGMENTATION_FUNCTIONS = {
    "Rotate": rotate,
    "Flip Horizontal": flip_horizontal,
    "Flip Vertical": flip_vertical,
    "Color Jitter": color_jitter,
    "Gaussian Blur": gaussian_blur,
    # Add more mappings here...
}

# This dictionary describes the parameters each technique accepts for the frontend form
# 'type' can be 'number', 'boolean', 'range', etc.
# 'default' is the initial value
# 'min', 'max', 'step' are for range/number inputs
AUGMENTATION_PARAMETERS = {
    "Rotate": [
        {"name": "degrees", "type": "number", "default": 0, "min": -180, "max": 180, "step": 1},
        {"name": "expand", "type": "boolean", "default": False},
    ],
    "Flip Horizontal": [], # No parameters needed
    "Flip Vertical": [],   # No parameters needed
    "Color Jitter": [
        {"name": "brightness", "type": "range", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
        {"name": "contrast", "type": "range", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
        {"name": "color", "type": "range", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
        {"name": "sharpness", "type": "range", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
    ],
    "Gaussian Blur": [
        {"name": "radius", "type": "number", "default": 0, "min": 0, "max": 10, "step": 0.1},
    ],
    # Add parameters for more techniques here...
}