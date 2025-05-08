import random
import numpy as np
from PIL import Image

def apply_mixup(image, other_image, alpha=0.4):
    """Apply mixup augmentation between two images."""
    lam = np.random.beta(alpha, alpha)
    image_array = np.array(image).astype(np.float32)
    other_array = np.array(other_image.resize(image.size)).astype(np.float32)
    mixed_array = lam * image_array + (1 - lam) * other_array
    return Image.fromarray(mixed_array.astype(np.uint8))

def apply_cutmix(image, other_image):
    """Apply cutmix augmentation between two images."""
    img_array = np.array(image)
    other_array = np.array(other_image.resize(image.size))
    h, w, _ = img_array.shape
    lam = np.random.beta(1.0, 1.0)
    bbx1 = np.random.randint(0, w)
    bby1 = np.random.randint(0, h)
    bbx2 = np.clip(bbx1 + int(w * np.sqrt(1 - lam)), 0, w)
    bby2 = np.clip(bby1 + int(h * np.sqrt(1 - lam)), 0, h)
    img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
    return Image.fromarray(img_array)
