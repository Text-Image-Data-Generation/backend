from PIL import Image
from augmentations.geometric import (rotate_image, scale_image,
                                     translate_image, flip_horizontal,
                                     flip_vertical, crop_image, pad_image)
from augmentations.color import (adjust_brightness, adjust_contrast,
                                 convert_grayscale, adjust_saturation)
from augmentations.noise import (add_gaussian_noise, add_salt_pepper_noise,
                                 add_speckle_noise, add_motion_blur)
from augmentations.occlusion import apply_cutout, apply_random_erasing
from augmentations.mix import apply_mixup, apply_cutmix

def apply_augmentations(image, techniques, params, uploaded_files):
    """Apply a series of augmentations based on selected techniques and parameters."""
    # Geometric Transformations
    if "rotation" in techniques and "rotation_angle" in params:
        angle = float(params["rotation_angle"])
        image = rotate_image(image, angle)

    if "scaling" in techniques and "scaling_factor" in params:
        scale = float(params["scaling_factor"])
        image = scale_image(image, scale)

    if "translation" in techniques and "translation_x" in params and "translation_y" in params:
        x_offset = int(params["translation_x"])
        y_offset = int(params["translation_y"])
        image = translate_image(image, x_offset, y_offset)

    if "flipping_horizontal" in techniques:
        image = flip_horizontal(image)

    if "flipping_vertical" in techniques:
        image = flip_vertical(image)

    if "cropping" in techniques:
        left = int(params.get("crop_left", 0))
        top = int(params.get("crop_top", 0))
        right = int(params.get("crop_right", 0))
        bottom = int(params.get("crop_bottom", 0))
        image = crop_image(image, left, top, right, bottom)

    if "padding" in techniques:
        padding = int(params.get("padding_size", 0))
        padding_color = params.get("padding_color", "#000000")
        image = pad_image(image, padding, padding_color)

    # Color Transformations
    if "brightness" in techniques and "brightness_factor" in params:
        image = adjust_brightness(image, float(params["brightness_factor"]))

    if "contrast" in techniques and "contrast_factor" in params:
        image = adjust_contrast(image, float(params["contrast_factor"]))

    if "grayscale" in techniques:
        image = convert_grayscale(image)

    if "saturation" in techniques and "saturation_factor" in params:
        image = adjust_saturation(image, float(params["saturation_factor"]))

    # Noise Transformations
    if "gaussian_noise" in techniques:
        var = float(params.get("gaussian_variance", 0.01))
        image = add_gaussian_noise(image, var=var)

    if "salt_pepper_noise" in techniques:
        amount = float(params.get("sap_amount", 0.005))
        image = add_salt_pepper_noise(image, amount=amount)

    if "speckle_noise" in techniques:
        image = add_speckle_noise(image)

    if "motion_blur" in techniques:
        size = int(params.get("motion_blur_size", 9))
        image = add_motion_blur(image, size=size)

    # Occlusion Transformations
    if "cutout" in techniques:
        size = int(params.get("cutout_size", 50))
        image = apply_cutout(image, size)

    if "random_erasing" in techniques:
        image = apply_random_erasing(image)

    # Mixup and Cutmix (requires another image from the uploaded set)
    if ("mixup" in techniques or "cutmix" in techniques) and uploaded_files:
        import random
        from flask import current_app
        other_filename = random.choice(uploaded_files)
        other_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], other_filename)
        other_image = Image.open(other_filepath)
        if "mixup" in techniques:
            alpha = float(params.get("mixup_alpha", 0.4))
            image = apply_mixup(image, other_image, alpha)
        if "cutmix" in techniques:
            image = apply_cutmix(image, other_image)

    return image
