from PIL import ImageEnhance, ImageOps

def adjust_brightness(image, factor):
    """Adjust brightness by the given factor."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """Adjust contrast by the given factor."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def convert_grayscale(image):
    """Convert image to grayscale."""
    return ImageOps.grayscale(image)

def adjust_saturation(image, factor):
    """Adjust saturation by the given factor."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)
