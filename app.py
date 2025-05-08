# # # # app.py - Full Flask Application with Image Augmentation API

# # # import os
# # # import json
# # # import random
# # # import zipfile
# # # import shutil
# # # import uuid
# # # import numpy as np # Needed for mixup/cutmix/noise
# # # import cv2 # Needed for motion blur
# # # from io import BytesIO
# # # from PIL import Image, ImageOps, ImageEnhance, ImageFilter # Pillow library
# # # from flask import Flask, request, jsonify, send_file
# # # from flask_cors import CORS

# # # # --- Configuration Variables ---

# # # UPLOAD_TEMP_FOLDER = 'uploaded_images_temp' # Temporary folder for uploads
# # # AUGMENTED_FOLDER = 'augmented_images'     # Folder for versioned augmented results
# # # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'} # Added bmp
# # # SECRET_KEY = 'your_very_secret_and_random_key_here' # CHANGE THIS IN PRODUCTION
# # # FRONTEND_URL = 'http://localhost:3000' # Default Create React App URL. Change if frontend runs elsewhere.

# # # # --- Utility Functions ---

# # # def allowed_file(filename):
# # #     """Check if the file extension is allowed."""
# # #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # # def get_temp_upload_path(upload_id):
# # #     """Gets the full path for a temporary upload folder."""
# # #     return os.path.join(UPLOAD_TEMP_FOLDER, upload_id)

# # # def cleanup_temp_upload(upload_id):
# # #     """Removes a temporary upload folder."""
# # #     temp_path = get_temp_upload_path(upload_id)
# # #     if os.path.exists(temp_path):
# # #         shutil.rmtree(temp_path)

# # # # --- Augmentation Functions ---
# # # # These functions take a PIL Image and parameters, and return a new PIL Image.

# # # # Color Transformations
# # # def adjust_brightness(image, factor=1.0):
# # #     """Adjust brightness by the given factor (0.0 to 2.0)."""
# # #     enhancer = ImageEnhance.Brightness(image)
# # #     return enhancer.enhance(factor)

# # # def adjust_contrast(image, factor=1.0):
# # #     """Adjust contrast by the given factor (0.0 to 2.0)."""
# # #     enhancer = ImageEnhance.Contrast(image)
# # #     return enhancer.enhance(factor)

# # # def convert_grayscale(image):
# # #     """Convert image to grayscale."""
# # #     # Convert to 'L' mode for grayscale
# # #     return ImageOps.grayscale(image)

# # # def adjust_saturation(image, factor=1.0):
# # #     """Adjust saturation by the given factor (0.0 to 2.0)."""
# # #     enhancer = ImageEnhance.Color(image)
# # #     return enhancer.enhance(factor)

# # # # Geometric Transformations
# # # def rotate_image(image, angle=0, expand=False):
# # #     """Rotate image by the given angle in degrees."""
# # #     return image.rotate(angle, expand=expand) # expand=True is important to not cut corners

# # # def scale_image(image, scale=1.0):
# # #     """Scale image by the given factor (e.g., 0.5 for half size, 2.0 for double)."""
# # #     if scale <= 0:
# # #         print("Warning: Scale factor must be positive. Using 1.0.")
# # #         scale = 1.0
# # #     new_size = (int(image.width * scale), int(image.height * scale))
# # #     # Ensure size is at least 1x1
# # #     new_size = (max(1, new_size[0]), max(1, new_size[1]))
# # #     # Use a high-quality resampling filter for scaling down
# # #     return image.resize(new_size, Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC)


# # # def translate_image(image, x_offset=0, y_offset=0, fill_color=(0, 0, 0)):
# # #     """Translate image by the given offsets (pixels). Fill empty areas with fill_color."""
# # #     # Calculate the size of the new image
# # #     # The new image needs to be large enough to contain the shifted image
# # #     # It should also accommodate negative offsets
# # #     new_width = image.width + abs(x_offset)
# # #     new_height = image.height + abs(y_offset)

# # #     # Create a new image of the calculated size with the fill color
# # #     new_image = Image.new(image.mode, (new_width, new_height), fill_color)

# # #     # Calculate the paste coordinates in the new image
# # #     # If x_offset is positive, paste starts at x_offset. If negative, paste starts at 0
# # #     # The image is shifted right by positive x, down by positive y
# # #     paste_x = max(0, x_offset)
# # #     paste_y = max(0, y_offset)

# # #     # Paste the original image into the new canvas
# # #     new_image.paste(image, (paste_x, paste_y))

# # #     # The translated image might be larger than the original.
# # #     # If the user wants to keep the original size, cropping would be needed,
# # #     # but translation usually implies changing the canvas size.
# # #     # We return the larger canvas. If original size is needed, add a crop step after.
# # #     return new_image


# # # def flip_horizontal(image):
# # #     """Flip image horizontally."""
# # #     return ImageOps.mirror(image)

# # # def flip_vertical(image):
# # #     """Flip image vertically."""
# # #     return ImageOps.flip(image)

# # # def crop_image(image, left=0, top=0, right=0, bottom=0):
# # #     """Crop image by pixels from each side."""
# # #     width, height = image.size
# # #     # Ensure crop boundaries are within image bounds and valid
# # #     x1 = max(0, left)
# # #     y1 = max(0, top)
# # #     x2 = min(width, width - right)
# # #     y2 = min(height, height - bottom)

# # #     if x2 <= x1 or y2 <= y1:
# # #         print("Warning: Invalid crop boundaries. Returning original image.")
# # #         return image # Return original if crop is invalid (e.g., cuts too much)

# # #     return image.crop((x1, y1, x2, y2))

# # # def pad_image(image, padding=0, fill_color=(0, 0, 0)):
# # #     """Add padding (border) around the image."""
# # #     if padding < 0:
# # #         print("Warning: Padding size cannot be negative. Using 0.")
# # #         padding = 0
# # #     return ImageOps.expand(image, border=padding, fill=fill_color)

# # # # Noise Transformations
# # # def add_gaussian_noise(image, mean=0, var=0.01):
# # #     """Add Gaussian noise to the image (var is variance, not std dev)."""
# # #     # Ensure image is in a suitable mode (like RGB) for numpy operations
# # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # #     sigma = var ** 0.5 # Convert variance to standard deviation
# # #     noise = np.random.normal(mean, sigma, img_array.shape)
# # #     img_noisy = img_array + noise
# # #     img_noisy = np.clip(img_noisy, 0, 1) # Clip values to be within [0, 1]
# # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # #     return Image.fromarray(img_noisy).convert(image.mode) # Convert back to original mode


# # # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# # #     """Add salt and pepper noise to the image."""
# # #     img_array = np.array(image.convert('RGB')) # Work with RGB array
# # #     total_pixels = img_array.size // img_array.shape[-1] # Account for channels
# # #     num_noise_pixels = int(amount * total_pixels)

# # #     # Salt noise
# # #     num_salt = int(num_noise_pixels * salt_vs_pepper)
# # #     coords = [np.random.randint(0, i, num_salt) for i in img_array.shape[:2]] # Only height and width
# # #     img_array[coords[0], coords[1], :] = 255 # Apply white noise across channels

# # #     # Pepper noise
# # #     num_pepper = num_noise_pixels - num_salt
# # #     coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape[:2]] # Only height and width
# # #     img_array[coords[0], coords[1], :] = 0 # Apply black noise across channels

# # #     return Image.fromarray(img_array).convert(image.mode) # Convert back to original mode

# # # def add_speckle_noise(image, mean=0, var=0.01):
# # #     """Add speckle noise to the image."""
# # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # #     sigma = var ** 0.5
# # #     noise = np.random.normal(mean, sigma, img_array.shape)
# # #     img_noisy = img_array + img_array * noise # Speckle noise formula
# # #     img_noisy = np.clip(img_noisy, 0, 1)
# # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # #     return Image.fromarray(img_noisy).convert(image.mode) # Convert back to original mode


# # # def add_motion_blur(image, size=9):
# # #     """Apply motion blur to the image using OpenCV."""
# # #     if size <= 1 or size % 2 == 0:
# # #         print("Warning: Motion blur size must be an odd integer > 1. Using default 9.")
# # #         size = 9 if size <= 1 or size % 2 == 0 else size # Ensure size is odd and > 1

# # #     img_array = np.array(image.convert('RGB')) # OpenCV works well with BGR/RGB numpy arrays

# # #     # Create the motion blur kernel
# # #     kernel = np.zeros((size, size), dtype=np.float32)
# # #     center = (size - 1) // 2
# # #     kernel[center, :] = 1.0 # Horizontal motion blur
# # #     kernel = kernel / size

# # #     # Apply the filter
# # #     img_blur_array = cv2.filter2D(img_array, -1, kernel)

# # #     return Image.fromarray(img_blur_array).convert(image.mode) # Convert back to original mode

# # # # Occlusion Transformations
# # # def apply_cutout(image, mask_size=50, fill_value=(0, 0, 0)):
# # #     """Apply cutout (a square filled with fill_value) to a random location."""
# # #     img_array = np.array(image.convert('RGB'))
# # #     h, w = img_array.shape[:2]

# # #     if mask_size <= 0:
# # #         print("Warning: Mask size must be positive. Using 50.")
# # #         mask_size = 50

# # #     if mask_size >= min(h, w):
# # #         print("Warning: Mask size is larger than image dimensions. Skipping cutout.")
# # #         return image # Cannot apply cutout if mask is too large

# # #     # Randomly select the center of the cutout
# # #     center_x = random.randint(0, w)
# # #     center_y = random.randint(0, h)

# # #     # Calculate the bounding box of the cutout
# # #     x1 = max(0, center_x - mask_size // 2)
# # #     y1 = max(0, center_y - mask_size // 2)
# # #     x2 = min(w, center_x + mask_size // 2)
# # #     y2 = min(h, center_y + mask_size // 2)

# # #     # Fill the cutout area
# # #     img_array[y1:y2, x1:x2] = fill_value # Apply fill value across channels

# # #     return Image.fromarray(img_array).convert(image.mode)

# # # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3, fill_value=(0, 0, 0)):
# # #     """Apply random erasing (a rectangle filled with fill_value or random noise) to a random region."""
# # #     img_array = np.array(image.convert('RGB'))
# # #     h, w = img_array.shape[:2]
# # #     area = h * w

# # #     # Ensure parameters are valid
# # #     if sl > sh or r1 > 1/r1 or not (0 <= sl < sh <= 1) or not (0 < r1 < 1/r1):
# # #          print("Warning: Invalid random erasing parameters. Skipping.")
# # #          return image

# # #     for _ in range(10): # Try a few times to find a valid erasing region
# # #         target_area = random.uniform(sl, sh) * area
# # #         aspect_ratio = random.uniform(r1, 1/r1)

# # #         erase_w = int(round(np.sqrt(target_area * aspect_ratio)))
# # #         erase_h = int(round(np.sqrt(target_area / aspect_ratio)))

# # #         if erase_w < w and erase_h < h:
# # #             erase_x = random.randint(0, w - erase_w)
# # #             erase_y = random.randint(0, h - erase_h)

# # #             # Decide fill type: random noise or solid color
# # #             # For simplicity, let's use solid color based on fill_value param
# # #             img_array[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = fill_value

# # #             return Image.fromarray(img_array).convert(image.mode)

# # #     # If after several attempts, no valid region is found, return original image
# # #     return Image.fromarray(img_array).convert(image.mode)


# # # # Mix Augmentations (require another image)
# # # def apply_mixup(image, other_image, alpha=0.4):
# # #     """Apply mixup augmentation between two images using a beta distribution."""
# # #     if image.size != other_image.size:
# # #         other_image = other_image.resize(image.size) # Resize other_image if needed

# # #     # Ensure images are in a compatible mode (like RGB)
# # #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# # #     other_array = np.array(other_image.convert('RGB')).astype(np.float32)

# # #     # Ensure alpha is valid
# # #     if alpha <= 0:
# # #         print("Warning: Mixup alpha must be positive. Using 0.4.")
# # #         alpha = 0.4

# # #     lam = np.random.beta(alpha, alpha)
# # #     mixed_array = lam * image_array + (1 - lam) * other_array
# # #     mixed_array = np.clip(mixed_array, 0, 255) # Clip values

# # #     return Image.fromarray(mixed_array.astype(np.uint8)).convert(image.mode)


# # # def apply_cutmix(image, other_image):
# # #     """Apply cutmix augmentation between two images by cutting and pasting a region."""
# # #     if image.size != other_image.size:
# # #         other_image = other_image.resize(image.size)

# # #     img_array = np.array(image.convert('RGB'))
# # #     other_array = np.array(other_image.convert('RGB'))

# # #     h, w, _ = img_array.shape
# # #     lam = np.random.beta(1.0, 1.0) # Use default alpha=beta=1.0 as per standard cutmix
# # #     cut_ratio_sq = 1.0 - lam # The proportion of the image to cut/paste

# # #     # Calculate bounding box size based on lambda
# # #     cut_w = int(w * np.sqrt(cut_ratio_sq))
# # #     cut_h = int(h * np.sqrt(cut_ratio_sq))

# # #     # Randomly select the center of the cut region
# # #     center_x = random.randint(0, w)
# # #     center_y = random.randint(0, h)

# # #     # Calculate the bounding box coordinates, ensuring they stay within image bounds
# # #     bbx1 = np.clip(center_x - cut_w // 2, 0, w)
# # #     bby1 = np.clip(center_y - cut_h // 2, 0, h)
# # #     bbx2 = np.clip(center_x + cut_w // 2, 0, w)
# # #     bby2 = np.clip(center_y + cut_h // 2, 0, h)

# # #     # Paste the cut region from the other image onto the current image
# # #     img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]

# # #     return Image.fromarray(img_array).convert(image.mode)


# # # # --- Information for the Frontend ---
# # # # This dictionary maps technique names (strings) to their corresponding functions
# # # AUGMENTATION_FUNCTIONS = {
# # #     # Color
# # #     "Adjust Brightness": adjust_brightness,
# # #     "Adjust Contrast": adjust_contrast,
# # #     "Convert Grayscale": convert_grayscale,
# # #     "Adjust Saturation": adjust_saturation,

# # #     # Geometric
# # #     "Rotate": rotate_image,
# # #     "Scale": scale_image,
# # #     "Translate": translate_image,
# # #     "Flip Horizontal": flip_horizontal,
# # #     "Flip Vertical": flip_vertical,
# # #     "Crop": crop_image,
# # #     "Pad": pad_image,

# # #     # Noise
# # #     "Gaussian Noise": add_gaussian_noise,
# # #     "Salt and Pepper Noise": add_salt_pepper_noise,
# # #     "Speckle Noise": add_speckle_noise,
# # #     "Motion Blur": add_motion_blur,

# # #     # Occlusion
# # #     "Cutout": apply_cutout,
# # #     "Random Erasing": apply_random_erasing,

# # #     # Mix (Require another image)
# # #     "Mixup": apply_mixup,
# # #     "Cutmix": apply_cutmix,
# # # }

# # # # This dictionary describes the parameters each technique accepts for the frontend form
# # # # 'type' can be 'number', 'boolean', 'range', 'color', etc.
# # # # 'default' is the initial value
# # # # 'min', 'max', 'step' are for number/range inputs
# # # # 'info' is a brief description for the frontend
# # # AUGMENTATION_PARAMETERS = {
# # #     "Adjust Brightness": [
# # #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is black, 2.0 is double brightness."},
# # #     ],
# # #     "Adjust Contrast": [
# # #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is solid gray, 2.0 is increased contrast."},
# # #     ],
# # #     "Convert Grayscale": [],
# # #     "Adjust Saturation": [
# # #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is grayscale, 2.0 is increased saturation."},
# # #     ],

# # #     "Rotate": [
# # #         {"name": "angle", "type": "number", "default": 0, "min": -360, "max": 360, "step": 1, "info": "Rotation angle in degrees."},
# # #         {"name": "expand", "type": "boolean", "default": False, "info": "Expand the canvas to fit the entire rotated image."},
# # #     ],
# # #     "Scale": [
# # #         {"name": "scale", "type": "number", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "info": "Scaling factor. 1.0 is original size."},
# # #     ],
# # #     "Translate": [
# # #         {"name": "x_offset", "type": "number", "default": 0, "min": -200, "max": 200, "step": 1, "info": "Horizontal translation in pixels."},
# # #         {"name": "y_offset", "type": "number", "default": 0, "min": -200, "max": 200, "step": 1, "info": "Vertical translation in pixels."},
# # #         {"name": "fill_color", "type": "color", "default": "#000000", "info": "Color to fill empty areas."},
# # #     ],
# # #     "Flip Horizontal": [],
# # #     "Flip Vertical": [],
# # #     "Crop": [
# # #          {"name": "left", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the left."},
# # #          {"name": "top", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the top."},
# # #          {"name": "right", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the right."},
# # #          {"name": "bottom", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the bottom."},
# # #     ],
# # #     "Pad": [
# # #          {"name": "padding", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to add as padding around the image."},
# # #          {"name": "fill_color", "type": "color", "default": "#000000", "info": "Color of the padding."},
# # #     ],

# # #     "Gaussian Noise": [
# # #         {"name": "var", "type": "number", "default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Variance of the Gaussian noise."},
# # #     ],
# # #      "Salt and Pepper Noise": [
# # #         {"name": "amount", "type": "range", "default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Proportion of pixels affected by noise."},
# # #         {"name": "salt_vs_pepper", "type": "range", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "info": "Ratio of salt noise (white) vs pepper noise (black). 0.5 is equal."},
# # #     ],
# # #     "Speckle Noise": [
# # #         {"name": "var", "type": "number", "default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Variance of the speckle noise."},
# # #     ],
# # #     "Motion Blur": [
# # #         {"name": "size", "type": "number", "default": 9, "min": 3, "max": 25, "step": 2, "info": "Size of the motion blur kernel (must be odd)."},
# # #     ],

# # #     "Cutout": [
# # #         {"name": "mask_size", "type": "number", "default": 50, "min": 10, "max": 200, "step": 1, "info": "Size of the square cutout region in pixels."},
# # #          {"name": "fill_value", "type": "color", "default": "#000000", "info": "Color of the cutout region."},
# # #     ],
# # #     "Random Erasing": [
# # #         {"name": "sl", "type": "range", "default": 0.02, "min": 0.0, "max": 0.5, "step": 0.001, "info": "Minimum proportion of the image area to erase."},
# # #         {"name": "sh", "type": "range", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.001, "info": "Maximum proportion of the image area to erase."},
# # #         {"name": "r1", "type": "range", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "info": "Minimum aspect ratio of the erased region."},
# # #          {"name": "fill_value", "type": "color", "default": "#000000", "info": "Color to fill the erased region."},
# # #     ],

# # #     "Mixup": [
# # #          {"name": "alpha", "type": "number", "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Alpha parameter for the Beta distribution. Controls the strength of mixing."},
# # #     ],
# # #     "Cutmix": [], # No parameters needed
# # # }

# # # # --- Augmentation Pipeline ---

# # # def apply_augmentations_pipeline(image: Image.Image, selected_augmentations_with_params: dict, available_files_in_temp_folder: list, temp_upload_path: str) -> Image.Image:
# # #     """
# # #     Applies a series of specified augmentation techniques to an image with given parameters.

# # #     Args:
# # #         image: The input PIL Image.
# # #         selected_augmentations_with_params: A dictionary where keys are technique names (strings)
# # #                                             and values are dictionaries of parameters for that technique.
# # #                                             Only techniques present here will be applied.
# # #                                             Example: {'Rotate': {'angle': 90, 'expand': True}, 'GaussianBlur': {'radius': 2}}
# # #         available_files_in_temp_folder: A list of filenames available in the temporary upload folder.
# # #                                        Needed for 'Mixup' and 'Cutmix' to pick another image.
# # #         temp_upload_path: The path to the temporary upload folder. Needed to load other images.

# # #     Returns:
# # #         The augmented PIL Image.
# # #     """
# # #     augmented_image = image.copy() # Start with a copy

# # #     # Ensure image is in RGB mode before applying many augmentations
# # #     # Some augmentations implicitly convert, but explicit conversion avoids errors with certain PIL functions
# # #     original_mode = augmented_image.mode
# # #     if original_mode not in ['RGB', 'RGBA', 'L']: # Keep L (grayscale) for grayscale conversion
# # #          augmented_image = augmented_image.convert('RGB')


# # #     for tech_name, params in selected_augmentations_with_params.items():
# # #         # Ensure the technique exists and is marked as enabled (if frontend uses an 'enabled' flag)
# # #         if tech_name in AUGMENTATION_FUNCTIONS:
# # #              # Clone params and remove 'enabled' flag if present from frontend
# # #              tech_params = params.copy()
# # #              tech_enabled = tech_params.pop('enabled', True) # Assume enabled if flag not present

# # #              if not tech_enabled:
# # #                   continue # Skip if the technique is not enabled

# # #              augmentation_func = AUGMENTATION_FUNCTIONS[tech_name]

# # #              try:
# # #                  # Handle special cases like Mixup/Cutmix that need another image
# # #                  if tech_name in ["Mixup", "Cutmix"]:
# # #                      if len(available_files_in_temp_folder) > 0:
# # #                          # Pick a random *different* image if possible, or the same if only one exists
# # #                          other_filename = random.choice(available_files_in_temp_folder)
# # #                          other_filepath = os.path.join(temp_upload_path, other_filename)
# # #                          try:
# # #                              other_image = Image.open(other_filepath)
# # #                               # Apply the mix/cut augmentation
# # #                              augmented_image = augmentation_func(augmented_image, other_image, **tech_params)
# # #                          except Exception as e:
# # #                              print(f"Error loading or using other image for {tech_name}: {e}")
# # #                              # Continue without applying this specific augmentation
# # #                      else:
# # #                          print(f"Warning: Cannot apply {tech_name}. No other images available in the batch.")
# # #                  elif tech_name in ["Translate", "Pad", "Cutout", "Random Erasing"] and 'fill_color' in tech_params:
# # #                      # Handle hex color string conversion for fill_color params
# # #                      hex_color = tech_params['fill_color']
# # #                      try:
# # #                          # Convert hex string to RGB tuple (e.g., "#RRGGBB" -> (R, G, B))
# # #                          rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
# # #                          tech_params['fill_color'] = rgb_color
# # #                          augmented_image = augmentation_func(augmented_image, **tech_params)
# # #                      except ValueError:
# # #                          print(f"Warning: Invalid hex color format for {tech_name}. Using black.")
# # #                          tech_params['fill_color'] = (0, 0, 0)
# # #                          augmented_image = augmentation_func(augmented_image, **tech_params)
# # #                  else:
# # #                     # Apply the general augmentation function with unpacked parameters
# # #                     augmented_image = augmentation_func(augmented_image, **tech_params)

# # #              except Exception as e:
# # #                  print(f"Error applying {tech_name}: {e}")
# # #                  # Continue to the next augmentation even if one fails

# # #         else:
# # #             print(f"Warning: Unknown augmentation technique '{tech_name}' skipped.")

# # #     # Convert back to original mode if it wasn't RGB, RGBA, or L and it was converted earlier
# # #     if original_mode not in ['RGB', 'RGBA', 'L'] and augmented_image.mode in ['RGB', 'RGBA', 'L']:
# # #          try:
# # #               augmented_image = augmented_image.convert(original_mode)
# # #          except Exception as e:
# # #               print(f"Warning: Could not convert back to original mode {original_mode}: {e}. Returning as RGB.")
# # #               # Keep as RGB if conversion fails


# # #     return augmented_image


# # # # --- Flask App Setup ---

# # # app = Flask(__name__)
# # # app.secret_key = SECRET_KEY
# # # # Allow requests from the React frontend origin
# # # CORS(app, origins=[FRONTEND_URL])

# # # # Ensure directories exist
# # # os.makedirs(UPLOAD_TEMP_FOLDER, exist_ok=True)
# # # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)


# # # # --- API Endpoints ---

# # # @app.route('/api/upload', methods=['POST'])
# # # def upload_files():
# # #     """Handles single file, multiple file, and zip file uploads."""
# # #     if 'files' not in request.files:
# # #         return jsonify({"error": "No files part in the request"}), 400

# # #     files = request.files.getlist('files')
# # #     if not files:
# # #         return jsonify({"error": "No selected file"}), 400

# # #     upload_id = str(uuid.uuid4())
# # #     temp_upload_path = get_temp_upload_path(upload_id)
# # #     os.makedirs(temp_upload_path, exist_ok=True)

# # #     uploaded_file_names = []

# # #     for file in files:
# # #         if file.filename == '':
# # #             continue # Skip empty file parts

# # #         original_filename = file.filename # Keep original name for logging/zip check

# # #         if file and allowed_file(original_filename):
# # #             filepath = os.path.join(temp_upload_path, original_filename)

# # #             if original_filename.lower().endswith('.zip'):
# # #                 # Handle zip file
# # #                 try:
# # #                     # Save the zip file temporarily to disk first for easier processing
# # #                     temp_zip_path = os.path.join(temp_upload_path, original_filename)
# # #                     file.save(temp_zip_path)

# # #                     with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
# # #                         for entry in zip_ref.namelist():
# # #                             # Avoid directories and __MACOSX folders
# # #                             if not entry.endswith('/') and not entry.startswith('__MACOSX/'):
# # #                                 entry_filename = os.path.basename(entry)
# # #                                 if allowed_file(entry_filename):
# # #                                     # Construct safe target path within temp folder
# # #                                     target_path = os.path.join(temp_upload_path, entry_filename)
# # #                                     # Prevent zip traversal attacks (check if target path is within the intended temp folder)
# # #                                     real_target_path = os.path.realpath(target_path)
# # #                                     real_temp_path = os.path.realpath(temp_upload_path)
# # #                                     if not real_target_path.startswith(real_temp_path):
# # #                                         print(f"Skipping potentially malicious path in zip: {entry}")
# # #                                         continue

# # #                                     # Check for duplicate filenames after extraction - append a number if needed
# # #                                     base, ext = os.path.splitext(entry_filename)
# # #                                     counter = 1
# # #                                     unique_entry_filename = entry_filename
# # #                                     unique_target_path = target_path # Start with target_path
# # #                                     while unique_entry_filename in uploaded_file_names:
# # #                                         unique_entry_filename = f"{base}_{counter}{ext}"
# # #                                         counter += 1
# # #                                         unique_target_path = os.path.join(temp_upload_path, unique_entry_filename) # Update target path

# # #                                     # Extract and save
# # #                                     source = zip_ref.open(entry)
# # #                                     target = open(unique_target_path, "wb")
# # #                                     with source, target:
# # #                                         shutil.copyfileobj(source, target)

# # #                                     uploaded_file_names.append(unique_entry_filename)
# # #                                 else:
# # #                                      print(f"Skipping non-allowed file in zip: {entry}")
# # #                     # Clean up the temporary zip file
# # #                     os.remove(temp_zip_path)

# # #                 except zipfile.BadZipFile:
# # #                     print(f"Error processing bad zip file: {original_filename}")
# # #                     # Continue to next file if this one was a bad zip, don't stop the whole upload
# # #                 except Exception as e:
# # #                      print(f"Error processing zip file {original_filename}: {e}")
# # #                      # Continue to next file if zip processing fails
# # #             else:
# # #                 # Handle single image file
# # #                 # Check for duplicate filenames - append a number if needed
# # #                 filename = original_filename # Start with the original name
# # #                 base, ext = os.path.splitext(filename)
# # #                 counter = 1
# # #                 unique_filename = filename
# # #                 unique_filepath = filepath # Start with the initial filepath
# # #                 while unique_filename in uploaded_file_names:
# # #                     unique_filename = f"{base}_{counter}{ext}"
# # #                     counter += 1
# # #                     unique_filepath = os.path.join(temp_upload_path, unique_filename) # Update filepath

# # #                 file.save(unique_filepath)
# # #                 uploaded_file_names.append(unique_filename)
# # #         else:
# # #              print(f"Skipping disallowed file: {original_filename}")


# # #     if not uploaded_file_names:
# # #         cleanup_temp_upload(upload_id)
# # #         return jsonify({"error": "No valid image files found in upload or zip"}), 400


# # #     return jsonify({"upload_id": upload_id, "files": uploaded_file_names}), 200

# # # @app.route('/api/augmentations', methods=['GET'])
# # # def get_augmentations_info():
# # #     """Returns a list of available augmentation techniques and their parameters."""
# # #     # Structure the response nicely for the frontend
# # #     augmentations_info = []
# # #     for name in AUGMENTATION_FUNCTIONS.keys(): # Iterate through function names
# # #         params = AUGMENTATION_PARAMETERS.get(name, [])
# # #         # Add a default 'enabled' boolean parameter for frontend checklist
# # #         params_for_frontend = [{"name": "enabled", "type": "boolean", "default": False, "info": "Enable this augmentation."}] + params
# # #         func = AUGMENTATION_FUNCTIONS[name]
# # #         augmentations_info.append({
# # #             "name": name,
# # #             "parameters": params_for_frontend, # Send parameters including 'enabled'
# # #             "description": func.__doc__.strip() if func.__doc__ else "No description available."
# # #         })
# # #     return jsonify({"augmentations": augmentations_info}), 200


# # # @app.route('/api/process', methods=['POST'])
# # # def process_augmentations():
# # #     """
# # #     Receives augmentation instructions and processes images.
# # #     Expects JSON body with:
# # #     {
# # #       "upload_id": "...",
# # #       "files_to_process": ["file1.jpg", ...], # List of filenames from the upload_id temp folder
# # #       "count": N, # How many total augmented images to generate (randomly pick from files_to_process with replacement)
# # #       "augmentations": {
# # #         "Technique Name": { "enabled": true/false, "param1": value, ... }, # Note the 'enabled' flag
# # #         ...
# # #       }
# # #     }
# # #     """
# # #     data = request.get_json()
# # #     if not data:
# # #         return jsonify({"error": "Invalid JSON body"}), 400

# # #     upload_id = data.get('upload_id')
# # #     files_to_process_requested = data.get('files_to_process', []) # Filenames requested by frontend
# # #     count = data.get('count')
# # #     selected_augmentations_with_params = data.get('augmentations', {}) # augmentations object from frontend

# # #     if not upload_id:
# # #         return jsonify({"error": "upload_id is missing"}), 400

# # #     temp_upload_path = get_temp_upload_path(upload_id)
# # #     if not os.path.exists(temp_upload_path):
# # #         return jsonify({"error": "Invalid upload_id or temporary files expired"}), 404

# # #     # Validate files_to_process against files currently in the temp folder
# # #     available_files = [f for f in os.listdir(temp_upload_path) if allowed_file(f)]

# # #     if not files_to_process_requested:
# # #          # If no specific files listed, assume all available files are eligible
# # #          files_eligible_for_processing = available_files
# # #     else:
# # #         # Filter the requested files to process to ensure they exist in the temp folder
# # #         files_eligible_for_processing = [f for f in files_to_process_requested if f in available_files]

# # #     if not files_eligible_for_processing:
# # #          cleanup_temp_upload(upload_id)
# # #          return jsonify({"error": "No valid files selected or available for processing"}), 400

# # #     # Filter out disabled augmentations before processing
# # #     enabled_augmentations_with_params = {
# # #         tech_name: params for tech_name, params in selected_augmentations_with_params.items()
# # #         if tech_name in AUGMENTATION_FUNCTIONS and params.get('enabled', True)
# # #     }

# # #     if not enabled_augmentations_with_params:
# # #          cleanup_temp_upload(upload_id)
# # #          return jsonify({"error": "No augmentation techniques enabled"}), 400


# # #     # Determine how many total augmented images to generate
# # #     # The request asks for `count` *total augmented images*.
# # #     # We will randomly pick from `files_eligible_for_processing` `count` times *with replacement*.
# # #     num_images_to_generate = int(count) if count is not None and count > 0 else len(files_eligible_for_processing)

# # #     if num_images_to_generate <= 0:
# # #          cleanup_temp_upload(upload_id)
# # #          return jsonify({"error": "Number of images to generate must be positive"}), 400

# # #     # Randomly pick source files with replacement up to num_images_to_generate times
# # #     picked_source_files = random.choices(files_eligible_for_processing, k=num_images_to_generate)

# # #     # Create a new version folder for augmented images
# # #     existing_versions = [int(d.split('_')[-1])
# # #                          for d in os.listdir(AUGMENTED_FOLDER)
# # #                          if d.startswith('version_')]
# # #     version_number = max(existing_versions + [0]) + 1
# # #     version_folder_name = f"version_{version_number}"
# # #     version_folder_path = os.path.join(AUGMENTED_FOLDER, version_folder_name)
# # #     os.makedirs(version_folder_path, exist_ok=True)

# # #     total_augmented_images_generated = 0

# # #     # Apply augmentations for each *picked* source file
# # #     for i, filename in enumerate(picked_source_files):
# # #         filepath = os.path.join(temp_upload_path, filename)
# # #         if not os.path.exists(filepath):
# # #              print(f"Warning: Source file not found during processing: {filepath}")
# # #              continue

# # #         try:
# # #             # Open the image. Use a context manager to ensure file is closed.
# # #             with Image.open(filepath) as image:
# # #                  image = image.copy() # Work on a copy as augmentation functions might modify

# # #             # Apply augmentations using the pipeline
# # #             augmented_image = apply_augmentations_pipeline(
# # #                 image,
# # #                 enabled_augmentations_with_params,
# # #                 available_files, # Pass all available files for Mix/Cut
# # #                 temp_upload_path # Pass temp path to load other images
# # #             )

# # #             # Save the augmented image
# # #             file_root, file_ext = os.path.splitext(filename)
# # #             # Generate a unique name for the augmented image within this version
# # #             # Append a counter and UUID snippet for uniqueness, link back to original filename
# # #             augmented_filename = f"{file_root}_aug{i+1}_{uuid.uuid4().hex[:4]}{file_ext}"
# # #             augmented_filepath = os.path.join(version_folder_path, augmented_filename)

# # #             # Determine save format based on original extension, default to JPEG or PNG
# # #             save_format = 'PNG' # PNG supports transparency better than JPEG
# # #             if file_ext.lower() in ['.jpg', '.jpeg', '.bmp']:
# # #                 save_format = 'JPEG'
# # #                 # If saving as JPEG, convert RGBA to RGB to avoid errors
# # #                 if augmented_image.mode == 'RGBA':
# # #                      augmented_image = augmented_image.convert('RGB')
# # #             elif file_ext.lower() == '.gif':
# # #                  save_format = 'GIF' # PIL might struggle with augmented GIFs, but try

# # #             # Save the image, handling potential errors
# # #             try:
# # #                  augmented_image.save(augmented_filepath, format=save_format)
# # #                  total_augmented_images_generated += 1
# # #             except Exception as save_error:
# # #                  print(f"Error saving augmented image {augmented_filename}: {save_error}")


# # #         except Exception as e:
# # #             print(f"Error processing source file {filename} for augmentation: {e}")
# # #             # Continue to the next image even if one fails


# # #     # Save metadata about this augmentation version
# # #     metadata = {
# # #         "version_id": version_number,
# # #         "timestamp": os.path.getctime(version_folder_path) if os.path.exists(version_folder_path) else None, # Creation timestamp
# # #         "source_images_in_batch": available_files,
# # #         "files_eligible_for_processing": files_eligible_for_processing,
# # #         "total_source_images_picked": len(picked_source_files), # How many times source images were picked
# # #         "total_augmented_images_generated": total_augmented_images_generated, # How many were actually saved
# # #         "selected_augmentations": enabled_augmentations_with_params, # Save params used
# # #         "notes": f"Generated {total_augmented_images_generated} images (attempted {len(picked_source_files)}) in version {version_number} from upload_id {upload_id}."
# # #     }
# # #     metadata_path = os.path.join(version_folder_path, "metadata.json")
# # #     try:
# # #         with open(metadata_path, 'w') as f:
# # #             json.dump(metadata, f, indent=4)
# # #     except Exception as metadata_error:
# # #          print(f"Error saving metadata for version {version_number}: {metadata_error}")


# # #     # Clean up the temporary upload folder *after* processing
# # #     # Only clean up if we successfully created the version folder, otherwise keep for debugging bad uploads
# # #     if os.path.exists(version_folder_path):
# # #         cleanup_temp_upload(upload_id)
# # #     else:
# # #          print(f"Warning: Version folder {version_folder_path} was not created. Keeping temp folder {temp_upload_path}.")


# # #     if total_augmented_images_generated == 0:
# # #          # If no images were successfully generated, maybe remove the version folder
# # #          if os.path.exists(version_folder_path):
# # #               shutil.rmtree(version_folder_path)
# # #          return jsonify({"error": "Failed to generate any augmented images. Check server logs for details."}), 500


# # #     return jsonify({"version_id": version_number, "message": f"Augmentation complete. Generated {total_augmented_images_generated} images in version {version_number}."}), 200

# # # @app.route('/api/versions', methods=['GET'])
# # # def list_versions():
# # #     """Lists available augmented image versions."""
# # #     versions = []
# # #     # List directories in the augmented folder, filter for version folders
# # #     version_folders = [d for d in os.listdir(AUGMENTED_FOLDER)
# # #                        if os.path.isdir(os.path.join(AUGMENTED_FOLDER, d))
# # #                        and d.startswith('version_')]

# # #     for folder in version_folders:
# # #         try:
# # #             version_number_str = folder.split('_')[-1]
# # #             if not version_number_str.isdigit():
# # #                  print(f"Skipping non-numeric version folder: {folder}")
# # #                  continue
# # #             version_number = int(version_number_str)
# # #             version_folder_path = os.path.join(AUGMENTED_FOLDER, folder)

# # #             metadata_path = os.path.join(version_folder_path, "metadata.json")
# # #             metadata = {}
# # #             if os.path.exists(metadata_path):
# # #                 with open(metadata_path, 'r') as f:
# # #                     metadata = json.load(f)
# # #             else:
# # #                  # If metadata is missing, create basic info by counting files
# # #                  metadata = {
# # #                      "version_id": version_number,
# # #                      "total_augmented_images_generated": len([f for f in os.listdir(version_folder_path) if allowed_file(f)]),
# # #                      "selected_augmentations": {"Info Missing": {"enabled": True}},
# # #                      "notes": "Metadata file not found for this version."
# # #                  }
# # #                  # Try to get timestamp from folder creation
# # #                  try:
# # #                      metadata["timestamp"] = os.path.getctime(version_folder_path)
# # #                  except Exception:
# # #                       metadata["timestamp"] = None


# # #             versions.append({
# # #                 "id": version_number,
# # #                 "metadata": metadata
# # #             })
# # #         except Exception as e:
# # #             print(f"Error processing version folder {folder}: {e}")
# # #             # Continue processing other folders

# # #     # Sort versions by ID in descending order (most recent first)
# # #     sorted_versions = sorted(versions, key=lambda x: x.get('id', 0), reverse=True) # Use .get for safety

# # #     return jsonify({"versions": sorted_versions}), 200

# # # @app.route('/api/download/<int:version_id>', methods=['GET'])
# # # def download_zip(version_id):
# # #     """Generates and serves a zip file for a specific augmented version."""
# # #     version_folder_name = f"version_{version_id}"
# # #     version_folder_path = os.path.join(AUGMENTED_FOLDER, version_folder_name)

# # #     if not os.path.isdir(version_folder_path):
# # #         return jsonify({"error": f"Version {version_id} not found"}), 404

# # #     zip_filename = f"augmented_images_v{version_id}.zip"
# # #     memory_file = BytesIO()

# # #     try:
# # #         with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
# # #             # Walk through the version folder and add files to the zip
# # #             for root, _, files in os.walk(version_folder_path):
# # #                 for file in files:
# # #                     # Add image files and the metadata file
# # #                     if allowed_file(file) or file == "metadata.json":
# # #                         file_path = os.path.join(root, file)
# # #                         # Use relpath to avoid including the full server path in the zip
# # #                         arcname = os.path.relpath(file_path, version_folder_path)
# # #                         zipf.write(file_path, arcname=arcname)

# # #     except Exception as e:
# # #          print(f"Error creating zip for version {version_id}: {e}")
# # #          return jsonify({"error": "Error creating zip file"}), 500

# # #     memory_file.seek(0)

# # #     # Send the file
# # #     return send_file(
# # #         memory_file,
# # #         download_name=zip_filename,
# # #         as_attachment=True,
# # #         mimetype='application/zip'
# # #     )

# # # # --- Main Execution ---

# # # if __name__ == '__main__':
# # #     # Ensure directories exist on startup
# # #     os.makedirs(UPLOAD_TEMP_FOLDER, exist_ok=True)
# # #     os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # #     # In development, run with debug=True. For production, use a production WSGI server (like Gunicorn or uWSGI).
# # #     # debug=True should NEVER be used in production due to security risks.
# # #     print(f"Flask app running. Visit {FRONTEND_URL} in your browser (ensure your React app is also running).")
# # #     print(f"API available at http://localhost:5001/api/...")
# # #     app.run(debug=True, port=5001) # Running on port 5001 by default




# # # app.py - Full Flask Application with Image Augmentation API, CSV Upload Handling, and CTGAN Data Generation

# # import os
# # import json
# # import random
# # import zipfile
# # import shutil
# # import uuid
# # import numpy as np # Needed for mixup/cutmix/noise
# # import cv2 # Needed for motion blur
# # import csv # Needed for CSV processing
# # from io import BytesIO
# # from PIL import Image, ImageOps, ImageEnhance, ImageFilter # Pillow library
# # from flask import Flask, request, jsonify, send_file
# # from flask_cors import CORS

# # # Imports for CTGAN data generation
# # import pandas as pd
# # from ctgan import CTGANSynthesizer

# # # --- Configuration Variables ---

# # # Base temporary folder for uploads
# # UPLOAD_TEMP_BASE_FOLDER = 'uploaded_data_temp'
# # # Subfolders within the temporary upload ID folder
# # UPLOAD_TEMP_IMAGE_SUBFOLDER = 'images'
# # UPLOAD_TEMP_CSV_SUBFOLDER = 'csv'

# # AUGMENTED_FOLDER = 'augmented_images'     # Folder for versioned augmented results
# # ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'} # Allowed image extensions
# # ALLOWED_CSV_EXTENSIONS = {'csv'} # Allowed CSV extensions
# # SECRET_KEY = 'your_very_secret_and_random_key_here' # CHANGE THIS IN PRODUCTION
# # FRONTEND_URL = 'http://localhost:3000' # Default Create React App URL. Change if frontend runs elsewhere.

# # # --- Utility Functions ---

# # def allowed_image_file(filename):
# #     """Check if the file extension is an allowed image extension."""
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# # def allowed_csv_file(filename):
# #     """Check if the file extension is an allowed CSV extension."""
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

# # def get_temp_upload_path(upload_id):
# #     """Gets the full path for a temporary upload ID folder."""
# #     return os.path.join(UPLOAD_TEMP_BASE_FOLDER, upload_id)

# # def get_temp_image_folder(upload_id):
# #     """Gets the path for the temporary images subfolder."""
# #     return os.path.join(get_temp_upload_path(upload_id), UPLOAD_TEMP_IMAGE_SUBFOLDER)

# # def get_temp_csv_folder(upload_id):
# #     """Gets the path for the temporary CSV subfolder."""
# #     return os.path.join(get_temp_upload_path(upload_id), UPLOAD_TEMP_CSV_SUBFOLDER)


# # def cleanup_temp_upload(upload_id):
# #     """Removes a temporary upload ID folder and its contents."""
# #     temp_path = get_temp_upload_path(upload_id)
# #     if os.path.exists(temp_path):
# #         shutil.rmtree(temp_path)
# #         print(f"Cleaned up temporary upload folder: {temp_path}")

# # # --- Augmentation Functions ---
# # # These functions take a PIL Image and parameters, and return a new PIL Image.

# # # Color Transformations
# # def adjust_brightness(image, factor=1.0):
# #     """Adjust brightness by the given factor (0.0 to 2.0)."""
# #     enhancer = ImageEnhance.Brightness(image)
# #     return enhancer.enhance(factor)

# # def adjust_contrast(image, factor=1.0):
# #     """Adjust contrast by the given factor (0.0 to 2.0)."""
# #     enhancer = ImageEnhance.Contrast(image)
# #     return enhancer.enhance(factor)

# # def convert_grayscale(image):
# #     """Convert image to grayscale."""
# #     return ImageOps.grayscale(image)

# # def adjust_saturation(image, factor=1.0):
# #     """Adjust saturation by the given factor (0.0 to 2.0)."""
# #     enhancer = ImageEnhance.Color(image)
# #     return enhancer.enhance(factor)

# # # Geometric Transformations
# # def rotate_image(image, angle=0, expand=False):
# #     """Rotate image by the given angle in degrees."""
# #     return image.rotate(angle, expand=expand)

# # def scale_image(image, scale=1.0):
# #     """Scale image by the given factor (e.g., 0.5 for half size, 2.0 for double)."""
# #     if scale <= 0:
# #         print("Warning: Scale factor must be positive. Using 1.0.")
# #         scale = 1.0
# #     new_size = (int(image.width * scale), int(image.height * scale))
# #     new_size = (max(1, new_size[0]), max(1, new_size[1])) # Ensure size is at least 1x1
# #     return image.resize(new_size, Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC)


# # def translate_image(image, x_offset=0, y_offset=0, fill_color=(0, 0, 0)):
# #     """Translate image by the given offsets (pixels). Fill empty areas with fill_color."""
# #     new_width = image.width + abs(x_offset)
# #     new_height = image.height + abs(y_offset)
# #     new_image = Image.new(image.mode, (new_width, new_height), fill_color)
# #     paste_x = max(0, x_offset)
# #     paste_y = max(0, y_offset)
# #     new_image.paste(image, (paste_x, paste_y))
# #     return new_image


# # def flip_horizontal(image):
# #     """Flip image horizontally."""
# #     return ImageOps.mirror(image)

# # def flip_vertical(image):
# #     """Flip image vertically."""
# #     return ImageOps.flip(image)

# # def crop_image(image, left=0, top=0, right=0, bottom=0):
# #     """Crop image by pixels from each side."""
# #     width, height = image.size
# #     x1 = max(0, left)
# #     y1 = max(0, top)
# #     x2 = min(width, width - right)
# #     y2 = min(height, height - bottom)
# #     if x2 <= x1 or y2 <= y1:
# #         print("Warning: Invalid crop boundaries. Returning original image.")
# #         return image
# #     return image.crop((x1, y1, x2, y2))

# # def pad_image(image, padding=0, fill_color=(0, 0, 0)):
# #     """Add padding (border) around the image."""
# #     if padding < 0:
# #         print("Warning: Padding size cannot be negative. Using 0.")
# #         padding = 0
# #     return ImageOps.expand(image, border=padding, fill=fill_color)

# # # Noise Transformations
# # def add_gaussian_noise(image, mean=0, var=0.01):
# #     """Add Gaussian noise to the image (var is variance, not std dev)."""
# #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# #     sigma = var ** 0.5
# #     noise = np.random.normal(mean, sigma, img_array.shape)
# #     img_noisy = img_array + noise
# #     img_noisy = np.clip(img_noisy, 0, 1)
# #     img_noisy = (img_noisy * 255).astype(np.uint8)
# #     return Image.fromarray(img_noisy).convert(image.mode)


# # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# #     """Add salt and pepper noise to the image."""
# #     img_array = np.array(image.convert('RGB'))
# #     total_pixels = img_array.size // img_array.shape[-1]
# #     num_noise_pixels = int(amount * total_pixels)

# #     num_salt = int(num_noise_pixels * salt_vs_pepper)
# #     coords = [np.random.randint(0, i, num_salt) for i in img_array.shape[:2]]
# #     img_array[coords[0], coords[1], :] = 255

# #     num_pepper = num_noise_pixels - num_salt
# #     coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape[:2]]
# #     img_array[coords[0], coords[1], :] = 0

# #     return Image.fromarray(img_array).convert(image.mode)

# # def add_speckle_noise(image, mean=0, var=0.01):
# #     """Add speckle noise to the image."""
# #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# #     sigma = var ** 0.5
# #     noise = np.random.normal(mean, sigma, img_array.shape)
# #     img_noisy = img_array + img_array * noise
# #     img_noisy = np.clip(img_noisy, 0, 1)
# #     img_noisy = (img_noisy * 255).astype(np.uint8)
# #     return Image.fromarray(img_noisy).convert(image.mode)


# # def add_motion_blur(image, size=9):
# #     """Apply motion blur to the image using OpenCV."""
# #     if size <= 1 or size % 2 == 0:
# #         print("Warning: Motion blur size must be an odd integer > 1. Using default 9.")
# #         size = 9 if size <= 1 or size % 2 == 0 else size
# #     img_array = np.array(image.convert('RGB'))
# #     kernel = np.zeros((size, size), dtype=np.float32)
# #     center = (size - 1) // 2
# #     kernel[center, :] = 1.0
# #     kernel = kernel / size
# #     img_blur_array = cv2.filter2D(img_array, -1, kernel)
# #     return Image.fromarray(img_blur_array).convert(image.mode)

# # # Occlusion Transformations
# # def apply_cutout(image, mask_size=50, fill_value=(0, 0, 0)):
# #     """Apply cutout (a square filled with fill_value) to a random location."""
# #     img_array = np.array(image.convert('RGB'))
# #     h, w = img_array.shape[:2]
# #     if mask_size <= 0:
# #         print("Warning: Mask size must be positive. Using 50.")
# #         mask_size = 50
# #     if mask_size >= min(h, w):
# #         print("Warning: Mask size is larger than image dimensions. Skipping cutout.")
# #         return image
# #     center_x = random.randint(0, w)
# #     center_y = random.randint(0, h)
# #     x1 = max(0, center_x - mask_size // 2)
# #     y1 = max(0, center_y - mask_size // 2)
# #     x2 = min(w, center_x + mask_size // 2)
# #     y2 = min(h, center_y + mask_size // 2)
# #     img_array[y1:y2, x1:x2] = fill_value
# #     return Image.fromarray(img_array).convert(image.mode)

# # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3, fill_value=(0, 0, 0)):
# #     """Apply random erasing (a rectangle filled with fill_value or random noise) to a random region."""
# #     img_array = np.array(image.convert('RGB'))
# #     h, w = img_array.shape[:2]
# #     area = h * w
# #     if sl > sh or r1 > 1/r1 or not (0 <= sl < sh <= 1) or not (0 < r1 < 1/r1):
# #          print("Warning: Invalid random erasing parameters. Skipping.")
# #          return image
# #     for _ in range(10):
# #         target_area = random.uniform(sl, sh) * area
# #         aspect_ratio = random.uniform(r1, 1/r1)
# #         erase_w = int(round(np.sqrt(target_area * aspect_ratio)))
# #         erase_h = int(round(np.sqrt(target_area / aspect_ratio)))
# #         if erase_w < w and erase_h < h:
# #             erase_x = random.randint(0, w - erase_w)
# #             erase_y = random.randint(0, h - erase_h)
# #             img_array[y_e:y_e+h_e, x_e:x_e+w_e] = fill_value # Assuming fill_value is RGB tuple
# #             return Image.fromarray(img_array).convert(image.mode)
# #     return Image.fromarray(img_array).convert(image.mode)


# # # Mix Augmentations (require another image)
# # def apply_mixup(image, other_image, alpha=0.4):
# #     """Apply mixup augmentation between two images using a beta distribution."""
# #     if image.size != other_image.size:
# #         other_image = other_image.resize(image.size)
# #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# #     other_array = np.array(other_image.convert('RGB')).astype(np.float32)
# #     if alpha <= 0:
# #         print("Warning: Mixup alpha must be positive. Using 0.4.")
# #         alpha = 0.4
# #     lam = np.random.beta(alpha, alpha)
# #     mixed_array = lam * image_array + (1 - lam) * other_array
# #     mixed_array = np.clip(mixed_array, 0, 255)
# #     return Image.fromarray(mixed_array.astype(np.uint8)).convert(image.mode)


# # def apply_cutmix(image, other_image):
# #     """Apply cutmix augmentation between two images by cutting and pasting a region."""
# #     if image.size != other_image.size:
# #         other_image = other_image.resize(image.size)
# #     img_array = np.array(image.convert('RGB'))
# #     other_array = np.array(other_image.convert('RGB'))
# #     h, w, _ = img_array.shape
# #     lam = np.random.beta(1.0, 1.0)
# #     cut_ratio_sq = 1.0 - lam
# #     cut_w = int(w * np.sqrt(cut_ratio_sq))
# #     cut_h = int(h * np.sqrt(cut_ratio_sq))
# #     center_x = random.randint(0, w)
# #     center_y = random.randint(0, h)
# #     bbx1 = np.clip(center_x - cut_w // 2, 0, w)
# #     bby1 = np.clip(center_y - cut_h // 2, 0, h)
# #     bbx2 = np.clip(center_x + cut_w // 2, 0, w)
# #     bby2 = np.clip(center_y + cut_h // 2, 0, h)
# #     img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
# #     return Image.fromarray(img_array).convert(image.mode)


# # # --- Information for the Frontend ---
# # # This dictionary maps technique names (strings) to their corresponding functions
# # AUGMENTATION_FUNCTIONS = {
# #     # Color
# #     "Adjust Brightness": adjust_brightness,
# #     "Adjust Contrast": adjust_contrast,
# #     "Convert Grayscale": convert_grayscale,
# #     "Adjust Saturation": adjust_saturation,

# #     # Geometric
# #     "Rotate": rotate_image,
# #     "Scale": scale_image,
# #     "Translate": translate_image,
# #     "Flip Horizontal": flip_horizontal,
# #     "Flip Vertical": flip_vertical,
# #     "Crop": crop_image,
# #     "Pad": pad_image,

# #     # Noise
# #     "Gaussian Noise": add_gaussian_noise,
# #     "Salt and Pepper Noise": add_salt_pepper_noise,
# #     "Speckle Noise": add_speckle_noise,
# #     "Motion Blur": add_motion_blur,

# #     # Occlusion
# #     "Cutout": apply_cutout,
# #     "Random Erasing": apply_random_erasing,

# #     # Mix (Require another image)
# #     "Mixup": apply_mixup,
# #     "Cutmix": apply_cutmix,
# # }

# # # This dictionary describes the parameters each technique accepts for the frontend form
# # # 'type' can be 'number', 'boolean', 'range', 'color', etc.
# # # 'default' is the initial value
# # # 'min', 'max', 'step' are for number/range inputs
# # # 'info' is a brief description for the frontend
# # AUGMENTATION_PARAMETERS = {
# #     "Adjust Brightness": [
# #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is black, 2.0 is double brightness."},
# #     ],
# #     "Adjust Contrast": [
# #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is solid gray, 2.0 is increased contrast."},
# #     ],
# #     "Convert Grayscale": [],
# #     "Adjust Saturation": [
# #         {"name": "factor", "type": "range", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Factor: 1.0 is original, 0.0 is grayscale, 2.0 is increased saturation."},
# #     ],

# #     "Rotate": [
# #         {"name": "angle", "type": "number", "default": 0, "min": -360, "max": 360, "step": 1, "info": "Rotation angle in degrees."},
# #         {"name": "expand", "type": "boolean", "default": False, "info": "Expand the canvas to fit the entire rotated image."},
# #     ],
# #     "Scale": [
# #         {"name": "scale", "type": "number", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "info": "Scaling factor. 1.0 is original size."},
# #     ],
# #     "Translate": [
# #         {"name": "x_offset", "type": "number", "default": 0, "min": -200, "max": 200, "step": 1, "info": "Horizontal translation in pixels."},
# #         {"name": "y_offset", "type": "number", "default": 0, "min": -200, "max": 200, "step": 1, "info": "Vertical translation in pixels."},
# #         {"name": "fill_color", "type": "color", "default": "#000000", "info": "Color to fill empty areas."},
# #     ],
# #     "Flip Horizontal": [],
# #     "Flip Vertical": [],
# #     "Crop": [
# #          {"name": "left", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the left."},
# #          {"name": "top", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the top."},
# #          {"name": "right", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the right."},
# #          {"name": "bottom", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to crop from the bottom."},
# #     ],
# #     "Pad": [
# #          {"name": "padding", "type": "number", "default": 0, "min": 0, "step": 1, "info": "Pixels to add as padding around the image."},
# #          {"name": "fill_color", "type": "color", "default": "#000000", "info": "Color of the padding."},
# #     ],

# #     "Gaussian Noise": [
# #         {"name": "var", "type": "number", "default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Variance of the Gaussian noise."},
# #     ],
# #      "Salt and Pepper Noise": [
# #         {"name": "amount", "type": "range", "default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Proportion of pixels affected by noise."},
# #         {"name": "salt_vs_pepper", "type": "range", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "info": "Ratio of salt noise (white) vs pepper noise (black). 0.5 is equal."},
# #     ],
# #     "Speckle Noise": [
# #         {"name": "var", "type": "number", "default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "info": "Variance of the speckle noise."},
# #     ],
# #     "Motion Blur": [
# #         {"name": "size", "type": "number", "default": 9, "min": 3, "max": 25, "step": 2, "info": "Size of the motion blur kernel (must be odd)."},
# #     ],

# #     "Cutout": [
# #         {"name": "mask_size", "type": "number", "default": 50, "min": 10, "max": 200, "step": 1, "info": "Size of the square cutout region in pixels."},
# #          {"name": "fill_value", "type": "color", "default": "#000000", "info": "Color of the cutout region."},
# #     ],
# #     "Random Erasing": [
# #         {"name": "sl", "type": "range", "default": 0.02, "min": 0.0, "max": 0.5, "step": 0.001, "info": "Minimum proportion of the image area to erase."},
# #         {"name": "sh", "type": "range", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.001, "info": "Maximum proportion of the image area to erase."},
# #         {"name": "r1", "type": "range", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "info": "Minimum aspect ratio of the erased region."},
# #          {"name": "fill_value", "type": "color", "default": "#000000", "info": "Color to fill the erased region."},
# #     ],

# #     "Mixup": [
# #          {"name": "alpha", "type": "number", "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01, "info": "Alpha parameter for the Beta distribution. Controls the strength of mixing."},
# #     ],
# #     "Cutmix": [], # No parameters needed
# # }

# # # --- Augmentation Pipeline ---

# # def apply_augmentations_pipeline(image: Image.Image, selected_augmentations_with_params: dict, available_image_files: list, temp_image_folder_path: str) -> Image.Image:
# #     """
# #     Apply a series of specified augmentation techniques to an image with given parameters.

# #     Args:
# #         image: The input PIL Image.
# #         selected_augmentations_with_params: A dictionary where keys are technique names (strings)
# #                                             and values are dictionaries of parameters for that technique.
# #                                             Only techniques present here will be applied.
# #         available_image_files: A list of image filenames available in the temporary image folder.
# #                                Needed for 'Mixup' and 'Cutmix' to pick another image.
# #         temp_image_folder_path: The path to the temporary image folder. Needed to load other images.

# #     Returns:
# #         The augmented PIL Image.
# #     """
# #     augmented_image = image.copy() # Start with a copy

# #     # Ensure image is in RGB mode before applying many augmentations
# #     original_mode = augmented_image.mode
# #     if original_mode not in ['RGB', 'RGBA', 'L']:
# #          augmented_image = augmented_image.convert('RGB')


# #     for tech_name, params in selected_augmentations_with_params.items():
# #         if tech_name in AUGMENTATION_FUNCTIONS:
# #              tech_params = params.copy()
# #              # The frontend might send an 'enabled' flag, remove it before passing to function
# #              tech_params.pop('enabled', None)

# #              augmentation_func = AUGMENTATION_FUNCTIONS[tech_name]

# #              try:
# #                  # Handle special cases like Mixup/Cutmix that need another image
# #                  if tech_name in ["Mixup", "Cutmix"]:
# #                      if len(available_image_files) > 0:
# #                          # Pick a random other image from the available images
# #                          other_filename = random.choice(available_image_files)
# #                          other_filepath = os.path.join(temp_image_folder_path, other_filename)
# #                          try:
# #                              with Image.open(other_filepath) as other_image:
# #                                   other_image_copy = other_image.copy() # Work on a copy
# #                              # Apply the mix/cut augmentation
# #                              augmented_image = augmentation_func(augmented_image, other_image_copy, **tech_params)
# #                          except Exception as e:
# #                              print(f"Error loading or using other image for {tech_name}: {e}")
# #                              # Continue without applying this specific augmentation
# #                      else:
# #                          print(f"Warning: Cannot apply {tech_name}. No other images available in the batch.")
# #                  elif tech_name in ["Translate", "Pad", "Cutout", "Random Erasing"] and 'fill_color' in tech_params:
# #                      # Handle hex color string conversion for fill_color params
# #                      hex_color = tech_params['fill_color']
# #                      try:
# #                          # Convert hex string to RGB tuple (e.g., "#RRGGBB" -> (R, G, B))
# #                          rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
# #                          tech_params['fill_color'] = rgb_color
# #                          augmented_image = augmentation_func(augmented_image, **tech_params)
# #                      except ValueError:
# #                          print(f"Warning: Invalid hex color format for {tech_name}: {hex_color}. Using black.")
# #                          tech_params['fill_color'] = (0, 0, 0)
# #                          augmented_image = augmentation_func(augmented_image, **tech_params)
# #                  else:
# #                     # Apply the general augmentation function with unpacked parameters
# #                     augmented_image = augmentation_func(augmented_image, **tech_params)

# #              except Exception as e:
# #                  print(f"Error applying {tech_name}: {e}")
# #                  # Continue to the next augmentation even if one fails

# #         else:
# #             print(f"Warning: Unknown augmentation technique '{tech_name}' skipped.")

# #     # Convert back to original mode if needed
# #     if original_mode not in ['RGB', 'RGBA', 'L'] and augmented_image.mode in ['RGB', 'RGBA', 'L']:
# #          try:
# #               augmented_image = augmented_image.convert(original_mode)
# #          except Exception as e:
# #               print(f"Warning: Could not convert back to original mode {original_mode}: {e}. Returning as RGB.")
# #               # Keep as RGB if conversion fails


# #     return augmented_image


# # # --- Flask App Setup ---

# # app = Flask(__name__)
# # app.secret_key = SECRET_KEY
# # # Allow requests from the React frontend origin
# # CORS(app, origins=[FRONTEND_URL])

# # # Ensure base temporary and augmented directories exist
# # os.makedirs(UPLOAD_TEMP_BASE_FOLDER, exist_ok=True)
# # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)


# # # --- API Endpoints (Image Augmentation) ---

# # @app.route('/api/upload', methods=['POST'])
# # def upload_files():
# #     """
# #     Handles image files (single, multiple, or in a zip) and a single CSV file upload.
# #     Expects FormData with fields like 'image_files' (for images/zip) and 'csv_file' (for CSV).
# #     """
# #     upload_id = str(uuid.uuid4())
# #     temp_upload_path = get_temp_upload_path(upload_id)
# #     temp_image_folder = get_temp_image_folder(upload_id)
# #     temp_csv_folder = get_temp_csv_folder(upload_id)

# #     # Create the temporary folders for this upload ID
# #     os.makedirs(temp_image_folder, exist_ok=True)
# #     os.makedirs(temp_csv_folder, exist_ok=True)

# #     uploaded_image_names = []
# #     uploaded_csv_name = None # Assume only one CSV file for simplicity

# #     # --- Handle Image Files ---
# #     if 'image_files' in request.files:
# #         image_files = request.files.getlist('image_files')
# #         for file in image_files:
# #             if file.filename == '':
# #                 continue

# #             original_filename = file.filename

# #             if file and (allowed_image_file(original_filename) or original_filename.lower().endswith('.zip')):
# #                  # Handle zip files containing images
# #                 if original_filename.lower().endswith('.zip'):
# #                     try:
# #                         # Save the zip file temporarily
# #                         temp_zip_path = os.path.join(temp_upload_path, original_filename)
# #                         file.save(temp_zip_path)

# #                         with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
# #                             for entry in zip_ref.namelist():
# #                                 if not entry.endswith('/') and not entry.startswith('__MACOSX/'):
# #                                     entry_filename = os.path.basename(entry)
# #                                     if allowed_image_file(entry_filename):
# #                                         # Construct safe target path within the images subfolder
# #                                         target_path = os.path.join(temp_image_folder, entry_filename)
# #                                         # Prevent zip traversal attacks
# #                                         real_target_path = os.path.realpath(target_path)
# #                                         real_image_folder = os.path.realpath(temp_image_folder)
# #                                         if not real_target_path.startswith(real_image_folder):
# #                                             print(f"Skipping potentially malicious path in zip: {entry}")
# #                                             continue

# #                                         # Check for duplicate filenames - append a number if needed
# #                                         base, ext = os.path.splitext(entry_filename)
# #                                         counter = 1
# #                                         unique_entry_filename = entry_filename
# #                                         unique_target_path = target_path
# #                                         while unique_entry_filename in uploaded_image_names:
# #                                             unique_entry_filename = f"{base}_{counter}{ext}"
# #                                             counter += 1
# #                                             unique_target_path = os.path.join(temp_image_folder, unique_entry_filename)

# #                                         # Extract and save to the images subfolder
# #                                         source = zip_ref.open(entry)
# #                                         target = open(unique_target_path, "wb")
# #                                         with source, target:
# #                                             shutil.copyfileobj(source, target)

# #                                         uploaded_image_names.append(unique_entry_filename)
# #                                     else:
# #                                          print(f"Skipping non-allowed file in zip: {entry}")
# #                         # Clean up the temporary zip file
# #                         os.remove(temp_zip_path)

# #                     except zipfile.BadZipFile:
# #                         print(f"Error processing bad zip file: {original_filename}")
# #                     except Exception as e:
# #                          print(f"Error processing zip file {original_filename}: {e}")
# #                 # Handle single image file
# #                 elif allowed_image_file(original_filename):
# #                     filename = original_filename
# #                     base, ext = os.path.splitext(filename)
# #                     counter = 1
# #                     unique_filename = filename
# #                     unique_filepath = os.path.join(temp_image_folder, unique_filename)
# #                     while unique_filename in uploaded_image_names:
# #                         unique_filename = f"{base}_{counter}{ext}"
# #                         counter += 1
# #                         unique_filepath = os.path.join(temp_image_folder, unique_filename)

# #                     # Save to the images subfolder
# #                     file.save(unique_filepath)
# #                     uploaded_image_names.append(unique_filename)
# #                 else:
# #                      print(f"Skipping disallowed image file: {original_filename}")


# #     # --- Handle CSV File ---
# #     if 'csv_file' in request.files:
# #         csv_file = request.files['csv_file']
# #         if csv_file.filename != '' and allowed_csv_file(csv_file.filename):
# #             # Assume only one CSV file is expected
# #             csv_filename = csv_file.filename
# #             csv_filepath = os.path.join(temp_csv_folder, csv_filename)
# #             try:
# #                  csv_file.save(csv_filepath)
# #                  uploaded_csv_name = csv_filename
# #             except Exception as e:
# #                  print(f"Error saving CSV file {csv_filename}: {e}")
# #                  # Continue without the CSV if saving fails


# #     if not uploaded_image_names and uploaded_csv_name is None:
# #         cleanup_temp_upload(upload_id)
# #         return jsonify({"error": "No valid image or CSV files found in upload"}), 400

# #     response_data = {
# #         "upload_id": upload_id,
# #         "image_files": uploaded_image_names,
# #         "csv_file": uploaded_csv_name # Will be null if no CSV was uploaded
# #     }

# #     return jsonify(response_data), 200

# # @app.route('/api/augmentations', methods=['GET'])
# # def get_augmentations_info():
# #     """Returns a list of available augmentation techniques and their parameters."""
# #     augmentations_info = []
# #     for name in AUGMENTATION_FUNCTIONS.keys():
# #         params = AUGMENTATION_PARAMETERS.get(name, [])
# #         params_for_frontend = [{"name": "enabled", "type": "boolean", "default": False, "info": "Enable this augmentation."}] + params
# #         func = AUGMENTATION_FUNCTIONS[name]
# #         augmentations_info.append({
# #             "name": name,
# #             "parameters": params_for_frontend,
# #             "description": func.__doc__.strip() if func.__doc__ else "No description available."
# #         })
# #     return jsonify({"augmentations": augmentations_info}), 200


# # @app.route('/api/process', methods=['POST'])
# # def process_augmentations():
# #     """
# #     Receives augmentation instructions and processes images.
# #     Expects JSON body with:
# #     {
# #       "upload_id": "...",
# #       "image_files_to_process": ["file1.jpg", ...], # List of image filenames from the upload_id temp folder
# #       "csv_file_name": "data.csv", # Optional: Name of the uploaded CSV file
# #       "count": N, # How many total augmented images to generate
# #       "augmentations": {
# #         "Technique Name": { "enabled": true/false, "param1": value, ... },
# #         ...
# #       }
# #     }
# #     """
# #     data = request.get_json()
# #     if not data:
# #         return jsonify({"error": "Invalid JSON body"}), 400

# #     upload_id = data.get('upload_id')
# #     image_files_to_process_requested = data.get('image_files_to_process', [])
# #     csv_file_name = data.get('csv_file_name') # Get the CSV filename
# #     count = data.get('count')
# #     selected_augmentations_with_params = data.get('augmentations', {})

# #     if not upload_id:
# #         return jsonify({"error": "upload_id is missing"}), 400

# #     temp_upload_path = get_temp_upload_path(upload_id)
# #     temp_image_folder = get_temp_image_folder(upload_id)
# #     temp_csv_folder = get_temp_csv_folder(upload_id)

# #     if not os.path.exists(temp_upload_path):
# #         return jsonify({"error": "Invalid upload_id or temporary data expired"}), 404

# #     # Validate image_files_to_process against files currently in the temp image folder
# #     available_image_files = [f for f in os.listdir(temp_image_folder) if allowed_image_file(f)]

# #     if not image_files_to_process_requested:
# #          # If no specific image files listed, assume all available images are eligible
# #          image_files_eligible_for_processing = available_image_files
# #     else:
# #         # Filter the requested image files to process to ensure they exist in the temp image folder
# #         image_files_eligible_for_processing = [f for f in image_files_to_process_requested if f in available_image_files]

# #     if not image_files_eligible_for_processing:
# #          # Clean up only if there were no eligible images to process
# #          cleanup_temp_upload(upload_id)
# #          return jsonify({"error": "No valid image files selected or available for processing"}), 400

# #     # --- CSV Processing (Optional) ---
# #     csv_data = None
# #     if csv_file_name:
# #         csv_filepath = os.path.join(temp_csv_folder, csv_file_name)
# #         if os.path.exists(csv_filepath):
# #             try:
# #                 with open(csv_filepath, mode='r', encoding='utf-8') as infile:
# #                     # Assuming the first row is header, subsequent rows are data
# #                     reader = csv.DictReader(infile)
# #                     csv_data = list(reader) # Read all rows into a list of dictionaries
# #                 print(f"Successfully loaded CSV data from {csv_file_name}. Rows: {len(csv_data)}")
# #                 # TODO: Implement logic to use csv_data for per-image parameters or selection
# #                 # For now, we just load it and print. The augmentation pipeline doesn't use it yet.
# #             except Exception as e:
# #                 print(f"Error reading or parsing CSV file {csv_file_name}: {e}")
# #                 # Continue processing without CSV data if there's an error
# #                 csv_file_name = None # Treat as if no valid CSV was provided
# #         else:
# #             print(f"Warning: Requested CSV file not found: {csv_filepath}")
# #             csv_file_name = None # Treat as if no valid CSV was provided


# #     # Filter out disabled augmentations
# #     enabled_augmentations_with_params = {
# #         tech_name: params for tech_name, params in selected_augmentations_with_params.items()
# #         if tech_name in AUGMENTATION_FUNCTIONS and params.get('enabled', True)
# #     }

# #     if not enabled_augmentations_with_params:
# #          # Clean up only if there were no enabled augmentations AND no images to process
# #          # (The check for image_files_eligible_for_processing already happened)
# #          cleanup_temp_upload(upload_id)
# #          return jsonify({"error": "No augmentation techniques enabled"}), 400


# #     # Determine how many total augmented images to generate
# #     num_images_to_generate = int(count) if count is not None and count > 0 else len(image_files_eligible_for_processing)

# #     if num_images_to_generate <= 0:
# #          cleanup_temp_upload(upload_id)
# #          return jsonify({"error": "Number of images to generate must be positive"}), 400

# #     # Randomly pick source image files with replacement up to num_images_to_generate times
# #     picked_source_image_files = random.choices(image_files_eligible_for_processing, k=num_images_to_generate)

# #     # Create a new version folder for augmented images
# #     existing_versions = [int(d.split('_')[-1])
# #                          for d in os.listdir(AUGMENTED_FOLDER)
# #                          if d.startswith('version_')]
# #     version_number = max(existing_versions + [0]) + 1
# #     version_folder_name = f"version_{version_number}"
# #     version_folder_path = os.path.join(AUGMENTED_FOLDER, version_folder_name)
# #     os.makedirs(version_folder_path, exist_ok=True)

# #     total_augmented_images_generated = 0

# #     # Apply augmentations for each *picked* source image file
# #     for i, filename in enumerate(picked_source_image_files):
# #         filepath = os.path.join(temp_image_folder, filename)
# #         if not os.path.exists(filepath):
# #              print(f"Warning: Source image file not found during processing: {filepath}")
# #              continue

# #         try:
# #             with Image.open(filepath) as image:
# #                  image = image.copy()

# #             # TODO: If CSV data is used for per-image parameters, retrieve them here
# #             # For now, we pass the same selected_augmentations_with_params to the pipeline
# #             # and the list of available image files for Mix/Cut.
# #             augmented_image = apply_augmentations_pipeline(
# #                 image,
# #                 enabled_augmentations_with_params,
# #                 available_image_files, # Pass list of all available images
# #                 temp_image_folder # Pass path to image folder
# #             )

# #             # Save the augmented image
# #             file_root, file_ext = os.path.splitext(filename)
# #             # Generate a unique name for the augmented image
# #             augmented_filename = f"{file_root}_aug{i+1}_{uuid.uuid4().hex[:4]}{file_ext}"
# #             augmented_filepath = os.path.join(version_folder_path, augmented_filename)

# #             save_format = 'PNG'
# #             if file_ext.lower() in ['.jpg', '.jpeg', '.bmp']:
# #                 save_format = 'JPEG'
# #                 if augmented_image.mode == 'RGBA':
# #                      augmented_image = augmented_image.convert('RGB')
# #             elif file_ext.lower() == '.gif':
# #                  save_format = 'GIF'

# #             try:
# #                  augmented_image.save(augmented_filepath, format=save_format)
# #                  total_augmented_images_generated += 1
# #             except Exception as save_error:
# #                  print(f"Error saving augmented image {augmented_filename}: {save_error}")


# #         except Exception as e:
# #             print(f"Error processing source image file {filename} for augmentation: {e}")


# #     # Save metadata about this augmentation version
# #     metadata = {
# #         "version_id": version_number,
# #         "timestamp": os.path.getctime(version_folder_path) if os.path.exists(version_folder_path) else None,
# #         "upload_id": upload_id, # Link back to the original upload batch
# #         "csv_file_used": csv_file_name, # Record which CSV was used (if any)
# #         "source_images_in_batch": available_image_files, # All images uploaded in this batch
# #         "image_files_eligible_for_processing": image_files_eligible_for_processing, # Images selected for processing
# #         "total_source_images_picked": len(picked_source_image_files),
# #         "total_augmented_images_generated": total_augmented_images_generated,
# #         "selected_augmentations": enabled_augmentations_with_params,
# #         "notes": f"Generated {total_augmented_images_generated} images (attempted {len(picked_source_image_files)}) in version {version_number} from upload_id {upload_id}."
# #     }
# #     metadata_path = os.path.join(version_folder_path, "metadata.json")
# #     try:
# #         with open(metadata_path, 'w') as f:
# #             json.dump(metadata, f, indent=4)
# #     except Exception as metadata_error:
# #          print(f"Error saving metadata for version {version_number}: {metadata_error}")


# #     # Clean up the temporary upload folder *after* processing
# #     # Only clean up if we successfully created the version folder, otherwise keep for debugging bad uploads
# #     if os.path.exists(version_folder_path):
# #         cleanup_temp_upload(upload_id)
# #     else:
# #          print(f"Warning: Version folder {version_folder_path} was not created. Keeping temp folder {temp_upload_path}.")


# #     if total_augmented_images_generated == 0:
# #          if os.path.exists(version_folder_path):
# #               shutil.rmtree(version_folder_path)
# #          # Check if CSV was uploaded even if no images were generated
# #          if csv_file_name and os.path.exists(os.path.join(temp_csv_folder, csv_file_name)):
# #              # If CSV exists but no images generated, maybe don't return 500 error?
# #              # Or return a specific error indicating image processing failed but CSV was fine.
# #              # For now, still return error if no images are generated as the primary task is image aug.
# #              pass # Keep the 500 error below if total_augmented_images_generated is 0


# #     if total_augmented_images_generated == 0:
# #          return jsonify({"error": "Failed to generate any augmented images. Check server logs for details."}), 500


# #     return jsonify({"version_id": version_number, "message": f"Augmentation complete. Generated {total_augmented_images_generated} images in version {version_number}."}), 200

# # @app.route('/api/versions', methods=['GET'])
# # def list_versions():
# #     """Lists available augmented image versions."""
# #     versions = []
# #     version_folders = [d for d in os.listdir(AUGMENTED_FOLDER)
# #                        if os.path.isdir(os.path.join(AUGMENTED_FOLDER, d))
# #                        and d.startswith('version_')]

# #     for folder in version_folders:
# #         try:
# #             version_number_str = folder.split('_')[-1]
# #             if not version_number_str.isdigit():
# #                  print(f"Skipping non-numeric version folder: {folder}")
# #                  continue
# #             version_number = int(version_number_str)
# #             version_folder_path = os.path.join(AUGMENTED_FOLDER, folder)

# #             metadata_path = os.path.join(version_folder_path, "metadata.json")
# #             metadata = {}
# #             if os.path.exists(metadata_path):
# #                 with open(metadata_path, 'r') as f:
# #                     metadata = json.load(f)
# #             else:
# #                  metadata = {
# #                      "version_id": version_number,
# #                      "total_augmented_images_generated": len([f for f in os.listdir(version_folder_path) if allowed_image_file(f)]),
# #                      "selected_augmentations": {"Info Missing": {"enabled": True}},
# #                      "notes": "Metadata file not found for this version."
# #                  }
# #                  try:
# #                      metadata["timestamp"] = os.path.getctime(version_folder_path)
# #                  except Exception:
# #                       metadata["timestamp"] = None

# #             versions.append({
# #                 "id": version_number,
# #                 "metadata": metadata
# #             })
# #         except Exception as e:
# #             print(f"Error processing version folder {folder}: {e}")

# #     sorted_versions = sorted(versions, key=lambda x: x.get('id', 0), reverse=True)

# #     return jsonify({"versions": sorted_versions}), 200

# # @app.route('/api/download/<int:version_id>', methods=['GET'])
# # def download_zip(version_id):
# #     """Generates and serves a zip file for a specific augmented version."""
# #     version_folder_name = f"version_{version_id}"
# #     version_folder_path = os.path.join(AUGMENTED_FOLDER, version_folder_name)

# #     if not os.path.isdir(version_folder_path):
# #         return jsonify({"error": f"Version {version_id} not found"}), 404

# #     zip_filename = f"augmented_images_v{version_id}.zip"
# #     memory_file = BytesIO()

# #     try:
# #         with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
# #             for root, _, files in os.walk(version_folder_path):
# #                 for file in files:
# #                     # Add image files and the metadata file
# #                     if allowed_image_file(file) or file == "metadata.json":
# #                         file_path = os.path.join(root, file)
# #                         arcname = os.path.relpath(file_path, version_folder_path)
# #                         zipf.write(file_path, arcname=arcname)

# #     except Exception as e:
# #          print(f"Error creating zip for version {version_id}: {e}")
# #          return jsonify({"error": "Error creating zip file"}), 500

# #     memory_file.seek(0)

# #     return send_file(
# #         memory_file,
# #         download_name=zip_filename,
# #         as_attachment=True,
# #         mimetype='application/zip'
# #     )

# # # --- API Endpoint (CTGAN Data Generation) ---

# # @app.route('/generate', methods=['POST'])
# # def generate_synthetic_data():
# #     """
# #     Generates synthetic data using CTGAN.
# #     Expects JSON body with:
# #     {
# #       "input_path": "path/to/input.csv_or_xlsx",
# #       "output_path": "path/to/output_directory",
# #       "n": 200, # number of samples to generate (optional, default 200)
# #       "epochs": 5 # number of training epochs (optional, default 5)
# #     }
# #     """
# #     data = request.json
# #     input_path = data.get('input_path')
# #     output_path = data.get('output_path')
# #     num_samples = int(data.get('n', 200))
# #     epochs = int(data.get('epochs', 5))

# #     if not input_path or not output_path:
# #         return jsonify({'error': 'input_path and output_path are required.'}), 400

# #     if not os.path.exists(input_path):
# #          return jsonify({'error': f'Input file not found at {input_path}.'}), 404

# #     # Ensure output directory exists
# #     os.makedirs(output_path, exist_ok=True)

# #     try:
# #         if input_path.lower().endswith('.csv'):
# #             df = pd.read_csv(input_path)
# #             output_filename = 'generated_data.csv'
# #         elif input_path.lower().endswith('.xlsx'):
# #             df = pd.read_excel(input_path)
# #             output_filename = 'generated_data.xlsx'
# #         else:
# #             return jsonify({'error': 'Unsupported file format for generation. Use .csv or .xlsx.'}), 400

# #         # Assuming all columns are discrete for simplicity as per original code
# #         discrete_columns = df.columns.tolist()

# #         # Initialize and fit CTGAN
# #         ctgan = CTGANSynthesizer(epochs=epochs)
# #         ctgan.fit(df, discrete_columns)

# #         # Generate synthetic samples
# #         samples = ctgan.sample(num_samples)

# #         # Determine output file path and save
# #         output_file = os.path.join(output_path, output_filename)
# #         if output_filename.endswith('.csv'):
# #              samples.to_csv(output_file, index=False)
# #         else: # .xlsx
# #              samples.to_excel(output_file, index=False)


# #         return jsonify({'output_file': output_file}), 200

# #     except FileNotFoundError:
# #          # This should be caught by the os.path.exists check, but good practice
# #          return jsonify({'error': f'Input file not found: {input_path}'}), 404
# #     except Exception as e:
# #         print(f"Error during CTGAN generation: {e}")
# #         return jsonify({'error': f'An error occurred during data generation: {e}'}), 500


# # # --- Main Execution ---

# # if __name__ == '__main__':
# #     # Ensure base directories exist on startup
# #     os.makedirs(UPLOAD_TEMP_BASE_FOLDER, exist_ok=True)
# #     os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# #     print(f"Flask app running. Visit {FRONTEND_URL} in your browser (ensure your React app is also running).")
# #     print(f"Image Augmentation API available at http://localhost:5001/api/...")
# #     print(f"CTGAN Generation API available at http://localhost:5001/generate")
# #     app.run(debug=True, port=5001)







# # app.py
# import os
# import json
# import random
# import zipfile
# from io import BytesIO
# from PIL import Image
# from flask import Flask, request, send_file, redirect, url_for, session, jsonify
# from flask_cors import CORS
# from werkzeug.utils import secure_filename

# from config import UPLOAD_FOLDER, AUGMENTED_FOLDER, SECRET_KEY, API_PORT, DEBUG
# from utils import allowed_file
# from augmentations.pipeline import apply_augmentations # Assuming augmentations folder is in the same directory or in PYTHONPATH

# app = Flask(__name__)
# app.secret_key = SECRET_KEY
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER
# app.config['API_PORT'] = API_PORT
# app.config['DEBUG'] = DEBUG

# # Enable CORS for all routes, allowing requests from your React frontend
# CORS(app, supports_credentials=True) # supports_credentials=True is important for sessions/cookies if used across origins

# # Ensure the upload and augmented directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # (Optional) MongoDB Integration Point:
# # from pymongo import MongoClient
# # if not DEBUG: # Or based on some other config
# #     client = MongoClient(app.config['MONGO_URI'])
# #     db = client[app.config['MONGO_DATABASE']]
# #     metadata_collection = db['augmentation_metadata']


# @app.route('/')
# def index():
#     # This route might not be directly used by the React app but is good for basic API health check.
#     return jsonify(message="Image Augmentation API is running.")

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """
#     Handles image uploads from the frontend.
#     Saves files to the UPLOAD_FOLDER and stores filenames in the session.
#     """
#     if 'images' not in request.files:
#         return jsonify(message="No image part in the request"), 400
    
#     files = request.files.getlist('images')
#     uploaded_filenames = []

#     if not files or files[0].filename == '':
#         return jsonify(message="No selected files"), 400

#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             try:
#                 file.save(filepath)
#                 uploaded_filenames.append(filename)
#             except Exception as e:
#                 app.logger.error(f"Failed to save file {filename}: {e}")
#                 return jsonify(message=f"Error saving file {filename}"), 500
#         elif file: # File present but not allowed
#             app.logger.warning(f"File type not allowed: {file.filename}")
#             # Optionally, inform about specific disallowed files, but for now, a general message
#             return jsonify(message=f"File type not allowed for {file.filename}. Allowed: {', '.join(list(app.config.get('ALLOWED_EXTENSIONS', [])))}"), 400


#     if not uploaded_filenames:
#          return jsonify(message="No valid files were uploaded or saved."), 400

#     session['uploaded_files'] = uploaded_filenames
#     app.logger.info(f"Uploaded files: {uploaded_filenames}")
#     # The frontend expects a success message or data to proceed.
#     # The session is set, and the frontend will move to the next step.
#     return jsonify(message=f"{len(uploaded_filenames)} files uploaded successfully", 
#                    uploaded_count=len(uploaded_filenames)), 200


# @app.route('/select_augmentations', methods=['POST'])
# def select_augmentations():
#     """
#     Receives selected augmentation techniques and parameters from the frontend.
#     Stores them in the session.
#     """
#     if request.method == 'POST':
#         # Frontend sends data as application/x-www-form-urlencoded, so request.form is correct
#         augmentations_data = request.form.to_dict()
#         if not augmentations_data:
#             return jsonify(message="No augmentation data received"), 400
        
#         session['augmentations'] = augmentations_data
#         app.logger.info(f"Selected augmentations: {augmentations_data}")
        
#         # The frontend expects a response that indicates success to move to the next step.
#         # The original code rendered a template here. For an API, we send JSON.
#         return jsonify(message="Augmentation selections saved", 
#                        selected_options=augmentations_data), 200
#     # GET request to this endpoint is not expected from the React app in this flow
#     return jsonify(message="Method not allowed for this endpoint in API mode"), 405


# @app.route('/set_augmentation_count', methods=['POST'])
# def set_augmentation_count():
#     """
#     Sets the number of images to augment, chosen by the user.
#     Stores this count in the session.
#     """
#     if 'uploaded_files' not in session or not session['uploaded_files']:
#         return jsonify(message="No files uploaded yet or session expired. Please upload images first."), 400

#     try:
#         images_to_augment = int(request.form.get('images_to_augment', 1))
#     except ValueError:
#         return jsonify(message="Invalid number for images_to_augment."), 400
        
#     if images_to_augment <= 0:
#         return jsonify(message="Number of images to augment must be positive."), 400

#     session['images_to_augment'] = images_to_augment
#     app.logger.info(f"Number of images to augment set to: {images_to_augment}")
#     # The frontend will proceed to call /apply_augmentations after this.
#     return jsonify(message=f"Number of images to augment set to {images_to_augment}"), 200


# @app.route('/apply_augmentations', methods=['GET']) # Changed to GET as per original logic, frontend calls this.
# def apply_augmentations_route():
#     """
#     Applies the selected augmentations to the specified number of uploaded images.
#     Saves augmented images in a new versioned folder and stores metadata.
#     """
#     uploaded_files = session.get('uploaded_files', [])
#     augmentations = session.get('augmentations', {})
#     images_to_augment_count = session.get('images_to_augment', 0)

#     if not uploaded_files:
#         return jsonify(message="No uploaded files found in session. Please start over."), 400
#     if not augmentations:
#         return jsonify(message="No augmentations selected in session. Please select augmentations."), 400
#     if images_to_augment_count <= 0:
#         return jsonify(message="Number of images to augment not set or invalid."), 400

#     # Prepare techniques list (only those checked 'yes' or present)
#     # The frontend sends params like `rotation: 'yes'`, `rotation_angle: '30'`
#     # We need to extract actual techniques and their parameters for the pipeline.
    
#     active_techniques = []
#     pipeline_params = {}
#     for key, value in augmentations.items():
#         # Check if the key represents an enabled technique (e.g., 'rotation': 'yes')
#         # Or if it's a parameter for an already implicitly enabled technique.
#         # The frontend sends all params, so we need to identify which techniques are "on".
#         # A simple way: if a technique 'foo' is 'yes', then 'foo' is active.
#         # All other keys are parameters.
#         if value == 'yes': # This was the convention in the original HTML forms.
#             active_techniques.append(key)
#         # All keys from the form are passed as params. The pipeline will pick what it needs.
#         pipeline_params[key] = value 
    
#     app.logger.info(f"Active techniques from session: {active_techniques}")
#     app.logger.info(f"Pipeline parameters from session: {pipeline_params}")


#     if not active_techniques:
#         return jsonify(message="No augmentation techniques were actively selected."), 400

#     # Create a new version folder for augmented images
#     try:
#         existing_versions = [int(d.split('_')[-1])
#                              for d in os.listdir(app.config['AUGMENTED_FOLDER'])
#                              if d.startswith('version_') and d.split('_')[-1].isdigit()]
#         version_number = max(existing_versions + [0]) + 1
#         version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], f"version_{version_number}")
#         os.makedirs(version_folder, exist_ok=True)
#     except Exception as e:
#         app.logger.error(f"Error creating version folder: {e}")
#         return jsonify(message="Server error creating augmentation version folder."), 500

#     total_augmented_images = 0
    
#     # Adjust count if requested images exceed available images
#     actual_images_to_augment_count = min(images_to_augment_count, len(uploaded_files))
    
#     if actual_images_to_augment_count == 0 and len(uploaded_files) > 0 : # If user set 0 but files exist, maybe augment 1? Or rely on validation.
#         # This case should be caught by previous checks or frontend validation.
#          actual_images_to_augment_count = 1 # Default to 1 if count is 0 but files exist (defensive)

#     if actual_images_to_augment_count == 0: # Still 0, means no uploaded files or bad count
#         return jsonify(message="No images available to augment or count is zero."), 400


#     images_to_process_list = random.sample(uploaded_files, actual_images_to_augment_count)
#     app.logger.info(f"Processing {len(images_to_process_list)} images: {images_to_process_list}")

#     processed_details = []

#     for filename in images_to_process_list:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         try:
#             image = Image.open(filepath)
#             # The pipeline needs the list of active techniques and all potential params
#             augmented_image = apply_augmentations(image.copy(), active_techniques, pipeline_params, uploaded_files)
            
#             file_root, file_ext = os.path.splitext(filename)
#             augmented_filename = f"{file_root}_aug_v{version_number}{file_ext}"
#             augmented_filepath = os.path.join(version_folder, augmented_filename)
#             augmented_image.save(augmented_filepath)
#             total_augmented_images += 1
#             processed_details.append({"original": filename, "augmented": augmented_filename, "status": "success"})
#         except FileNotFoundError:
#             app.logger.error(f"Uploaded file not found during augmentation: {filepath}")
#             processed_details.append({"original": filename, "status": "error", "message": "File not found"})
#         except Exception as e:
#             app.logger.error(f"Error augmenting image {filename}: {e}")
#             processed_details.append({"original": filename, "status": "error", "message": str(e)})

#     # Save metadata about this augmentation version
#     # (Optional) MongoDB Integration Point: Replace file-based metadata with MongoDB
#     metadata = {
#         "version": version_number,
#         "total_augmented_images": total_augmented_images,
#         "requested_images_to_augment": images_to_augment_count,
#         "actually_processed_count": len(images_to_process_list),
#         "selected_augmentations": active_techniques,
#         "augmentation_params_sent": pipeline_params, # Store what was sent to pipeline
#         "processed_files_details": processed_details
#     }
#     try:
#         with open(os.path.join(version_folder, "metadata.json"), 'w') as f:
#             json.dump(metadata, f, indent=4)
#     except Exception as e:
#         app.logger.error(f"Error saving metadata.json for version {version_number}: {e}")
#         # Continue, but log the error. The augmentation itself might have succeeded.

#     # Clear session data related to this batch and remove original uploaded images
#     session.pop('uploaded_files', None)
#     session.pop('augmentations', None)
#     session.pop('images_to_augment', None)
    
#     # Cleanup uploaded files
#     for file_to_remove in os.listdir(app.config['UPLOAD_FOLDER']):
#         try:
#             os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_to_remove))
#         except Exception as e:
#             app.logger.error(f"Error removing uploaded file {file_to_remove}: {e}")

#     # The frontend will call /download next. This endpoint signals completion.
#     return jsonify(message=f"Augmentation process for version {version_number} completed. {total_augmented_images} images augmented.",
#                    version=version_number,
#                    augmented_count=total_augmented_images), 200


# @app.route('/download', methods=['GET'])
# def download_versions_list():
#     """
#     Lists available augmented image versions and their metadata.
#     Returns data in JSON format for the frontend.
#     """
#     versions_data = {}
#     try:
#         version_folders = [d for d in os.listdir(app.config['AUGMENTED_FOLDER']) if d.startswith('version_')]
#     except FileNotFoundError:
#         app.logger.warning(f"Augmented folder not found: {app.config['AUGMENTED_FOLDER']}")
#         return jsonify(versions={}), 200 # Return empty if base folder doesn't exist

#     for folder_name in version_folders:
#         try:
#             version_number_str = folder_name.split('_')[-1]
#             if not version_number_str.isdigit():
#                 app.logger.warning(f"Skipping non-numeric version folder: {folder_name}")
#                 continue
#             version_number = int(version_number_str)
            
#             # (Optional) MongoDB Integration Point: Fetch metadata from MongoDB here
#             metadata_path = os.path.join(app.config['AUGMENTED_FOLDER'], folder_name, "metadata.json")
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'r') as f:
#                     metadata = json.load(f)
#             else: # Fallback if metadata.json is missing
#                 metadata = {
#                     "total_augmented_images": "N/A",
#                     "selected_augmentations": ["Unknown - metadata missing"],
#                     "augmentation_params": {},
#                     "error": "metadata.json not found"
#                 }
#             versions_data[version_number] = metadata
#         except Exception as e:
#             app.logger.error(f"Error reading metadata for {folder_name}: {e}")
#             versions_data[version_number] = {"error": f"Could not load metadata for version {version_number_str}"}

#     # Sort versions by version number, descending (newest first)
#     sorted_versions = dict(sorted(versions_data.items(), key=lambda item: item[0], reverse=True))
#     return jsonify(versions=sorted_versions)


# @app.route('/download_zip/<version_num_str>')
# def download_zip(version_num_str):
#     """
#     Creates a ZIP archive of a specific augmented image version and sends it for download.
#     Includes images and the metadata.json file.
#     """
#     if not version_num_str.isdigit():
#         return jsonify(message="Invalid version number format."), 400
    
#     version_folder_name = f"version_{version_num_str}"
#     version_path = os.path.join(app.config['AUGMENTED_FOLDER'], version_folder_name)

#     if not os.path.isdir(version_path):
#         return jsonify(message=f"Version {version_num_str} not found."), 404

#     zip_filename = f"augmented_images_version_{version_num_str}.zip"
#     memory_file = BytesIO()

#     try:
#         with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             for root, _, files in os.walk(version_path):
#                 for file in files:
#                     # Include common image types and the metadata file
#                     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.json')):
#                         file_path = os.path.join(root, file)
#                         # Arcname is the path inside the zip file
#                         arcname = os.path.relpath(file_path, version_path)
#                         zipf.write(file_path, arcname)
#         memory_file.seek(0)
#     except Exception as e:
#         app.logger.error(f"Error creating ZIP for version {version_num_str}: {e}")
#         return jsonify(message="Error creating ZIP file."), 500
        
#     return send_file(
#         memory_file,
#         download_name=zip_filename, # Changed from attachment_filename for newer Flask/Werkzeug
#         as_attachment=True,
#         mimetype='application/zip'
#     )

# # Global error handlers for common HTTP errors, returning JSON
# @app.errorhandler(404)
# def not_found_error(error):
#     return jsonify(message="Resource not found", error=str(error)), 404

# @app.errorhandler(500)
# def internal_error(error):
#     # Log the error for server-side debugging
#     app.logger.error(f"Server Error: {error}")
#     return jsonify(message="Internal server error", error="An unexpected error occurred on the server."), 500

# @app.errorhandler(400)
# def bad_request_error(error):
#     # This handler might be explicitly called or raised by Flask for malformed requests
#     return jsonify(message="Bad request", error=str(error.description if hasattr(error, 'description') else error)), 400

# @app.errorhandler(405)
# def method_not_allowed_error(error):
#     return jsonify(message="Method not allowed for the requested URL", error=str(error)), 405


# if __name__ == '__main__':
#     app.logger.info(f"Starting Flask app on port {app.config['API_PORT']}")
#     app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
#     app.logger.info(f"Augmented folder: {app.config['AUGMENTED_FOLDER']}")
#     app.run(host='0.0.0.0', port=app.config['API_PORT'], debug=app.config['DEBUG'])





# app.py
import os
import json
import random
import zipfile
from io import BytesIO
from PIL import Image
from flask import Flask, request, send_file, session, jsonify # Removed redirect, url_for as not used by API directly
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure config and utils are loaded correctly
from config import UPLOAD_FOLDER, AUGMENTED_FOLDER, SECRET_KEY, API_PORT, DEBUG, ALLOWED_EXTENSIONS
from utils import allowed_file
from augmentations.pipeline import apply_augmentations # Assuming augmentations folder is in the same directory or in PYTHONPATH

app = Flask(__name__)
app.secret_key = SECRET_KEY # Crucial: Use the static key from config.py
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER
app.config['API_PORT'] = API_PORT
app.config['DEBUG'] = DEBUG
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS # Make sure this is set for allowed_file

# Enable CORS for all routes, allowing requests from your React frontend
# origins="http://localhost:3000" can be used for more specificity if your React app is always on port 3000
CORS(app, supports_credentials=True, origins=["http://localhost:3000", "http://127.0.0.1:3000"] + (["http://127.0.0.1:5001".rsplit(":",1)[0] + ":3000"] if "http://127.0.0.1:5001" else []))


# Ensure the upload and augmented directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return jsonify(message="Image Augmentation API is running.")

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'images' not in request.files:
        return jsonify(message="No image part in the request"), 400
    
    files = request.files.getlist('images')
    uploaded_filenames = []
    errors = []

    if not files or files[0].filename == '': # Check if any file is actually selected
        return jsonify(message="No selected files"), 400

    for file in files:
        if file.filename == '': # Skip if a file input is empty
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                uploaded_filenames.append(filename)
            except Exception as e:
                app.logger.error(f"Failed to save file {filename}: {e}")
                errors.append(f"Error saving file {filename}")
        elif file: 
            app.logger.warning(f"File type not allowed: {file.filename}")
            errors.append(f"File type not allowed for {file.filename}. Allowed: {', '.join(list(app.config['ALLOWED_EXTENSIONS']))}")

    if not uploaded_filenames and errors: # Only errors, no successful uploads
        return jsonify(message="File upload failed.", errors=errors), 400
    if not uploaded_filenames and not errors: # No files processed (e.g. all empty filenames)
        return jsonify(message="No valid files were processed."),400


    session['uploaded_files'] = uploaded_filenames
    app.logger.info(f"Session after upload: {session.get('uploaded_files')}")
    return jsonify(message=f"{len(uploaded_filenames)} files uploaded successfully.", 
                   uploaded_count=len(uploaded_filenames), errors=errors), 200


@app.route('/select_augmentations', methods=['POST'])
def select_augmentations_route(): # Renamed to avoid conflict with imported function if any
    if request.method == 'POST':
        augmentations_data = request.form.to_dict()
        if not augmentations_data:
            return jsonify(message="No augmentation data received"), 400
        
        session['augmentations'] = augmentations_data
        app.logger.info(f"Session after select_augmentations: {session.get('augmentations')}")
        return jsonify(message="Augmentation selections saved", 
                       selected_options=augmentations_data), 200
    return jsonify(message="Method not allowed"), 405


@app.route('/set_augmentation_count', methods=['POST'])
def set_augmentation_count_route(): # Renamed
    app.logger.info(f"Session at start of set_augmentation_count: uploaded_files={session.get('uploaded_files')}")
    if 'uploaded_files' not in session or not session['uploaded_files']:
        app.logger.error("Error in set_augmentation_count: 'uploaded_files' not in session or empty.")
        return jsonify(message="No files uploaded yet or session expired. Please upload images first."), 400

    try:
        images_to_augment = int(request.form.get('images_to_augment', 1))
    except ValueError:
        return jsonify(message="Invalid number for images_to_augment."), 400
        
    if images_to_augment <= 0:
        return jsonify(message="Number of images to augment must be positive."), 400
    
    if images_to_augment > len(session['uploaded_files']):
        return jsonify(message=f"Cannot augment {images_to_augment} images, only {len(session['uploaded_files'])} available."), 400


    session['images_to_augment'] = images_to_augment
    app.logger.info(f"Session after set_augmentation_count: images_to_augment={session.get('images_to_augment')}")
    return jsonify(message=f"Number of images to augment set to {images_to_augment}"), 200


@app.route('/apply_augmentations', methods=['GET'])
def apply_augmentations_route():
    app.logger.info(f"Session at start of apply_augmentations: uploaded_files={session.get('uploaded_files')}, augmentations={session.get('augmentations')}, count={session.get('images_to_augment')}")
    uploaded_files = session.get('uploaded_files', [])
    augmentations_params_from_session = session.get('augmentations', {}) # These are raw from form
    images_to_augment_count = session.get('images_to_augment', 0)

    if not uploaded_files:
        return jsonify(message="No uploaded files found in session. Please start over by uploading images."), 400
    if not augmentations_params_from_session:
        return jsonify(message="No augmentations selected in session. Please select augmentations first."), 400
    if images_to_augment_count <= 0:
        return jsonify(message="Number of images to augment not set or invalid. Please set the count."), 400

    active_techniques = []
    pipeline_params = {} # Params for the augmentation functions
    
    # Extract techniques and their specific parameters
    # Example: augmentations_params_from_session = {'rotation': 'yes', 'rotation_angle': '30', 'brightness': 'yes', 'brightness_factor': '1.2'}
    for key, value in augmentations_params_from_session.items():
        if value == 'yes': # This identifies an active technique
            active_techniques.append(key)
        # All items from form are considered potential parameters
        pipeline_params[key] = value
    
    if not active_techniques:
        return jsonify(message="No augmentation techniques were actively selected (e.g., 'rotation': 'yes')."), 400

    app.logger.info(f"Active techniques for pipeline: {active_techniques}")
    app.logger.info(f"Parameters for pipeline: {pipeline_params}")

    try:
        existing_versions = [int(d.split('_')[-1])
                             for d in os.listdir(app.config['AUGMENTED_FOLDER'])
                             if d.startswith('version_') and d.split('_')[-1].isdigit()]
        version_number = max(existing_versions + [0]) + 1
        version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], f"version_{version_number}")
        os.makedirs(version_folder, exist_ok=True)
    except Exception as e:
        app.logger.error(f"Error creating version folder: {e}")
        return jsonify(message="Server error creating augmentation version folder."), 500

    total_augmented_images = 0
    actual_images_to_process_count = min(images_to_augment_count, len(uploaded_files))
    
    if actual_images_to_process_count == 0:
        return jsonify(message="Adjusted count of images to augment is zero. No images to process."), 400

    images_to_process_list = random.sample(uploaded_files, actual_images_to_process_count)
    processed_details = []

    for filename in images_to_process_list:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            image = Image.open(filepath)
            augmented_image = apply_augmentations(image.copy(), active_techniques, pipeline_params, uploaded_files) # Pass all params
            
            file_root, file_ext = os.path.splitext(filename)
            augmented_filename = f"{file_root}_aug_v{version_number}{file_ext}"
            augmented_filepath = os.path.join(version_folder, augmented_filename)
            augmented_image.save(augmented_filepath)
            total_augmented_images += 1
            processed_details.append({"original": filename, "augmented": augmented_filename, "status": "success"})
        except FileNotFoundError:
            app.logger.error(f"Uploaded file not found during augmentation: {filepath}")
            processed_details.append({"original": filename, "status": "error", "message": "File not found"})
        except Exception as e:
            app.logger.error(f"Error augmenting image {filename}: {e}")
            processed_details.append({"original": filename, "status": "error", "message": str(e)})

    metadata = {
        "version": version_number,
        "total_augmented_images": total_augmented_images,
        "requested_images_to_augment": images_to_augment_count,
        "actually_processed_count": len(images_to_process_list),
        "selected_augmentations": active_techniques, # List of techniques like ['rotation', 'brightness']
        "augmentation_params_sent": pipeline_params, # All params sent from frontend
        "processed_files_details": processed_details
    }
    try:
        with open(os.path.join(version_folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        app.logger.error(f"Error saving metadata.json for version {version_number}: {e}")

    # Clear session data for this batch
    session.pop('uploaded_files', None)
    session.pop('augmentations', None)
    session.pop('images_to_augment', None)
    app.logger.info("Session cleared after applying augmentations.")
    
    # Cleanup original uploaded files
    for file_to_remove in os.listdir(app.config['UPLOAD_FOLDER']):
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_to_remove))
        except Exception as e:
            app.logger.error(f"Error removing uploaded file {file_to_remove}: {e}")

    return jsonify(message=f"Augmentation process for version {version_number} completed. {total_augmented_images} images augmented.",
                   version=version_number,
                   augmented_count=total_augmented_images), 200


@app.route('/download', methods=['GET'])
def download_versions_list():
    versions_data = {}
    try:
        version_folders = [d for d in os.listdir(app.config['AUGMENTED_FOLDER']) if d.startswith('version_')]
    except FileNotFoundError:
        app.logger.warning(f"Augmented folder not found: {app.config['AUGMENTED_FOLDER']}")
        return jsonify(versions={}), 200

    for folder_name in version_folders:
        try:
            version_number_str = folder_name.split('_')[-1]
            if not version_number_str.isdigit():
                continue
            version_number = int(version_number_str)
            
            metadata_path = os.path.join(app.config['AUGMENTED_FOLDER'], folder_name, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else: 
                metadata = { "error": "metadata.json not found" }
            versions_data[version_number] = metadata
        except Exception as e:
            app.logger.error(f"Error reading metadata for {folder_name}: {e}")
            versions_data[version_number] = {"error": f"Could not load metadata for version {version_number_str}"}

    sorted_versions = dict(sorted(versions_data.items(), key=lambda item: item[0], reverse=True))
    return jsonify(versions=sorted_versions)


@app.route('/download_zip/<version_num_str>')
def download_zip_route(version_num_str): # Renamed
    if not version_num_str.isdigit():
        return jsonify(message="Invalid version number format."), 400
    
    version_folder_name = f"version_{version_num_str}"
    version_path = os.path.join(app.config['AUGMENTED_FOLDER'], version_folder_name)

    if not os.path.isdir(version_path):
        return jsonify(message=f"Version {version_num_str} not found."), 404

    zip_filename = f"augmented_images_version_{version_num_str}.zip"
    memory_file = BytesIO()

    try:
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(version_path):
                for file_in_zip in files: # Renamed to avoid conflict
                    if file_in_zip.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.json')):
                        file_path = os.path.join(root, file_in_zip)
                        arcname = os.path.relpath(file_path, version_path)
                        zipf.write(file_path, arcname)
        memory_file.seek(0)
    except Exception as e:
        app.logger.error(f"Error creating ZIP for version {version_num_str}: {e}")
        return jsonify(message="Error creating ZIP file."), 500
        
    return send_file(
        memory_file,
        download_name=zip_filename,
        as_attachment=True,
        mimetype='application/zip'
    )

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify(message="Resource not found", error=str(error)), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}", exc_info=True) # Log full traceback
    return jsonify(message="Internal server error", error="An unexpected error occurred on the server."), 500

@app.errorhandler(400)
def bad_request_error(error):
    description = str(error.description if hasattr(error, 'description') else error)
    app.logger.warning(f"Bad Request: {description}")
    return jsonify(message="Bad request", error=description), 400

@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify(message="Method not allowed for the requested URL", error=str(error)), 405


if __name__ == '__main__':
    app.logger.info(f"Starting Flask app on port {app.config['API_PORT']}")
    app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Augmented folder: {app.config['AUGMENTED_FOLDER']}")
    app.logger.info(f"DEBUG mode: {app.config['DEBUG']}")
    app.logger.info(f"SECRET_KEY is set (check config.py for its value)")
    app.run(host='0.0.0.0', port=app.config['API_PORT'], debug=app.config['DEBUG'])
