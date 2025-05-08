from PIL import Image, ImageOps

def rotate_image(image, angle):
    """Rotate image by the given angle."""
    return image.rotate(angle, expand=True)

def scale_image(image, scale):
    """Scale image by the given factor."""
    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)

def translate_image(image, x_offset, y_offset):
    """Translate image by the given offsets."""
    new_width = image.width + abs(x_offset)
    new_height = image.height + abs(y_offset)
    new_image = Image.new("RGB", (new_width, new_height), "black")
    new_image.paste(image, (max(0, x_offset), max(0, y_offset)))
    return new_image

def flip_horizontal(image):
    """Flip image horizontally."""
    return ImageOps.mirror(image)

def flip_vertical(image):
    """Flip image vertically."""
    return ImageOps.flip(image)

def crop_image(image, left, top, right, bottom):
    """Crop image with the given boundaries."""
    width, height = image.size
    return image.crop((left, top, width - right, height - bottom))

def pad_image(image, padding, padding_color):
    """Add padding around the image."""
    return ImageOps.expand(image, border=padding, fill=padding_color)
