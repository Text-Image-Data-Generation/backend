import numpy as np
from PIL import Image
import cv2

def add_gaussian_noise(image, mean=0, var=0.01):
    """Add Gaussian noise to the image."""
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(mean, var ** 0.5, img_array.shape)
    img_noisy = img_array + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    """Add salt and pepper noise to the image."""
    img_array = np.array(image)
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * img_array.size * (1.0 - salt_vs_pepper)).astype(int)

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape]
    img_array[tuple(coords)] = 255

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape]
    img_array[tuple(coords)] = 0

    return Image.fromarray(img_array)

def add_speckle_noise(image):
    """Add speckle noise to the image."""
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.randn(*img_array.shape)
    img_noisy = img_array + img_array * noise
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def add_motion_blur(image, size=9):
    """Apply motion blur to the image."""
    kernel = np.zeros((size, size))
    kernel[int((size - 1)/2), :] = np.ones(size)
    kernel = kernel / size
    img_array = np.array(image)
    img_blur = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(img_blur)
