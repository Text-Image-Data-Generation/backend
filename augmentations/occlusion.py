import numpy as np
from PIL import Image

def apply_cutout(image, mask_size):
    """Apply cutout (a black box) to a random location in the image."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)

    img_array[y1:y2, x1:x2] = 0  # Black box
    return Image.fromarray(img_array)

def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
    """Apply random erasing to a region of the image."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    s = np.random.uniform(sl, sh) * h * w
    r = np.random.uniform(r1, 1/r1)
    w_e = int(np.sqrt(s * r))
    h_e = int(np.sqrt(s / r))
    if w_e == 0 or h_e == 0:
        return Image.fromarray(img_array)
    x_e = np.random.randint(0, w - w_e)
    y_e = np.random.randint(0, h - h_e)
    img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, 3))
    return Image.fromarray(img_array)
