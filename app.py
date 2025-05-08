# # # <<<<<<< Updated upstream
# # # =======
# # # # # # # app.py
# # # # # # import os
# # # # # # import zipfile
# # # # # # import json
# # # # # # from flask import Flask, request, jsonify, send_from_directory
# # # # # # from flask_cors import CORS
# # # # # # from werkzeug.utils import secure_filename
# # # # # # from PIL import Image, ImageOps

# # # # # # app = Flask(__name__)
# # # # # # CORS(app)

# # # # # # UPLOAD_FOLDER = 'uploads'
# # # # # # AUGMENTED_FOLDER = 'augmented'
# # # # # # METADATA_FILE = 'augmentation_metadata.json'

# # # # # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # # # # app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # # # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # # # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # # # # # # Load augmentation metadata
# # # # # # if os.path.exists(METADATA_FILE):
# # # # # #     with open(METADATA_FILE, 'r') as f:
# # # # # #         augmentation_metadata = json.load(f)
# # # # # # else:
# # # # # #     augmentation_metadata = {}

# # # # # # # Augmentation functions
# # # # # # def rotate_image(image, angle=90):
# # # # # #     return image.rotate(angle, expand=True)

# # # # # # def scale_image(image, scale=1.5):
# # # # # #     width, height = image.size
# # # # # #     return image.resize((int(width * scale), int(height * scale)))

# # # # # # def flip_horizontal(image):
# # # # # #     return ImageOps.mirror(image)

# # # # # # def flip_vertical(image):
# # # # # #     return ImageOps.flip(image)

# # # # # # def apply_augmentations(image, augmentations):
# # # # # #     for aug in augmentations:
# # # # # #         if aug == 'rotate':
# # # # # #             image = rotate_image(image)
# # # # # #         elif aug == 'scale':
# # # # # #             image = scale_image(image)
# # # # # #         elif aug == 'flip_horizontal':
# # # # # #             image = flip_horizontal(image)
# # # # # #         elif aug == 'flip_vertical':
# # # # # #             image = flip_vertical(image)
# # # # # #     return image

# # # # # # @app.route('/upload', methods=['POST'])
# # # # # # def upload_files():
# # # # # #     dataset = request.form.get('dataset')
# # # # # #     files = request.files.getlist('files')

# # # # # #     if not dataset:
# # # # # #         return "No dataset name provided", 400

# # # # # #     dataset_folder = os.path.join(UPLOAD_FOLDER, secure_filename(dataset))
# # # # # #     os.makedirs(dataset_folder, exist_ok=True)

# # # # # #     for file in files:
# # # # # #         filename = secure_filename(file.filename)
# # # # # #         filepath = os.path.join(dataset_folder, filename)
# # # # # #         file.save(filepath)

# # # # # #         if filename.endswith('.zip'):
# # # # # #             with zipfile.ZipFile(filepath, 'r') as zip_ref:
# # # # # #                 zip_ref.extractall(dataset_folder)
# # # # # #             os.remove(filepath)

# # # # # #     return "Files uploaded", 200

# # # # # # @app.route('/datasets', methods=['GET'])
# # # # # # def list_datasets():
# # # # # #     datasets = []
# # # # # #     for folder in os.listdir(UPLOAD_FOLDER):
# # # # # #         folder_path = os.path.join(UPLOAD_FOLDER, folder)
# # # # # #         if os.path.isdir(folder_path):
# # # # # #             files = os.listdir(folder_path)
# # # # # #             metadata = augmentation_metadata.get(folder, {})
# # # # # #             datasets.append({
# # # # # #                 'name': folder,
# # # # # #                 'count': len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]),
# # # # # #                 'files': files,
# # # # # #                 'augmentations': metadata.get("augmentations", []),
# # # # # #                 'augmented_zip': metadata.get("augmented_zip")
# # # # # #             })
# # # # # #     return jsonify(datasets)

# # # # # # @app.route('/augment', methods=['POST'])
# # # # # # def augment_dataset():
# # # # # #     data = request.get_json()
# # # # # #     dataset_name = secure_filename(data.get('datasetName'))
# # # # # #     augmentations = data.get('augmentations', [])

# # # # # #     source_folder = os.path.join(UPLOAD_FOLDER, dataset_name)
# # # # # #     target_folder = os.path.join(AUGMENTED_FOLDER, dataset_name)
# # # # # #     os.makedirs(target_folder, exist_ok=True)

# # # # # #     for filename in os.listdir(source_folder):
# # # # # #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# # # # # #             img_path = os.path.join(source_folder, filename)
# # # # # #             with Image.open(img_path) as img:
# # # # # #                 augmented_img = apply_augmentations(img, augmentations)
# # # # # #                 save_path = os.path.join(target_folder, f"aug_{filename}")
# # # # # #                 augmented_img.save(save_path)

# # # # # #     # Zip the augmented folder
# # # # # #     zip_filename = f"{dataset_name}_augmented.zip"
# # # # # #     zip_path = os.path.join(target_folder, zip_filename)
# # # # # #     with zipfile.ZipFile(zip_path, 'w') as zipf:
# # # # # #         for file in os.listdir(target_folder):
# # # # # #             if file.endswith(('.png', '.jpg', '.jpeg')):
# # # # # #                 zipf.write(os.path.join(target_folder, file), file)

# # # # # #     # Save metadata
# # # # # #     augmentation_metadata[dataset_name] = {
# # # # # #         "augmentations": augmentations,
# # # # # #         "augmented_zip": zip_filename
# # # # # #     }
# # # # # #     with open(METADATA_FILE, 'w') as f:
# # # # # #         json.dump(augmentation_metadata, f)

# # # # # #     return jsonify({'message': 'Augmentation complete', 'zip': zip_filename})

# # # # # # @app.route('/uploads/<dataset>/<filename>')
# # # # # # def uploaded_file(dataset, filename):
# # # # # #     return send_from_directory(os.path.join(UPLOAD_FOLDER, secure_filename(dataset)), filename)

# # # # # # @app.route('/augmented/<dataset>/<filename>')
# # # # # # def augmented_file(dataset, filename):
# # # # # #     return send_from_directory(os.path.join(AUGMENTED_FOLDER, secure_filename(dataset)), filename)

# # # # # # from flask import send_from_directory

# # # # # # @app.route('/download/<dataset_name>/<filename>')
# # # # # # def download_file(dataset_name, filename):
# # # # # #     augmented_dir = os.path.join('augmented', dataset_name)
# # # # # #     return send_from_directory(augmented_dir, filename, as_attachment=True)


# # # # # # if __name__ == '__main__':
# # # # # #     app.run(host='0.0.0.0', port=5001, debug=True)





# # # # # import os
# # # # # import zipfile
# # # # # import json
# # # # # import random
# # # # # import numpy as np
# # # # # import cv2 # For motion blur

# # # # # from flask import Flask, request, jsonify, send_from_directory, current_app
# # # # # from flask_cors import CORS
# # # # # from werkzeug.utils import secure_filename
# # # # # from PIL import Image, ImageOps, ImageEnhance, ImageChops

# # # # # app = Flask(__name__)
# # # # # CORS(app)

# # # # # UPLOAD_FOLDER = 'uploads'
# # # # # AUGMENTED_FOLDER = 'augmented'
# # # # # METADATA_FILE = 'augmentation_metadata.json'

# # # # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # # # app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # # # # # Load augmentation metadata
# # # # # if os.path.exists(METADATA_FILE):
# # # # #     with open(METADATA_FILE, 'r') as f:
# # # # #         augmentation_metadata = json.load(f)
# # # # # else:
# # # # #     augmentation_metadata = {}

# # # # # # --- Start of Augmentation Functions ---

# # # # # # Geometric Transformations
# # # # # def rotate_image(image, angle=90):
# # # # #     return image.rotate(angle, expand=True)

# # # # # def scale_image(image, scale=1.5):
# # # # #     if scale <= 0: return image # Avoid invalid scale
# # # # #     width, height = image.size
# # # # #     return image.resize((int(width * scale), int(height * scale)))

# # # # # def translate_image(image, x_offset, y_offset):
# # # # #     return ImageChops.offset(image, x_offset, y_offset)

# # # # # def flip_horizontal(image):
# # # # #     return ImageOps.mirror(image)

# # # # # def flip_vertical(image):
# # # # #     return ImageOps.flip(image)

# # # # # def crop_image(image, left, top, right, bottom):
# # # # #     width, height = image.size
# # # # #     # Ensure crop box is within image dimensions and valid
# # # # #     left = max(0, int(left))
# # # # #     top = max(0, int(top))
# # # # #     right = min(width, int(right))
# # # # #     bottom = min(height, int(bottom))
# # # # #     if left < right and top < bottom:
# # # # #         return image.crop((left, top, right, bottom))
# # # # #     return image # Return original if crop dimensions are invalid


# # # # # def pad_image(image, padding_size, padding_color="#000000"):
# # # # #     padding_size = int(padding_size)
# # # # #     if padding_size <=0: return image
# # # # #     return ImageOps.expand(image, border=padding_size, fill=padding_color)

# # # # # # Color Transformations
# # # # # def adjust_brightness(image, factor):
# # # # #     enhancer = ImageEnhance.Brightness(image.convert('RGB')) # Ensure RGB for brightness
# # # # #     return enhancer.enhance(factor)

# # # # # def adjust_contrast(image, factor):
# # # # #     enhancer = ImageEnhance.Contrast(image.convert('RGB')) # Ensure RGB for contrast
# # # # #     return enhancer.enhance(factor)

# # # # # def convert_grayscale(image):
# # # # #     return ImageOps.grayscale(image).convert('RGB') # Often want to keep 3 channels for consistency

# # # # # def adjust_saturation(image, factor):
# # # # #     enhancer = ImageEnhance.Color(image.convert('RGB')) # Ensure RGB for saturation
# # # # #     return enhancer.enhance(factor)

# # # # # # Noise Transformations
# # # # # def add_gaussian_noise(image, mean=0, var=0.01):
# # # # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # # # #     noise = np.random.normal(mean, var ** 0.5, img_array.shape)
# # # # #     img_noisy = img_array + noise
# # # # #     img_noisy = np.clip(img_noisy, 0, 1)
# # # # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # # # #     return Image.fromarray(img_noisy)

# # # # # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# # # # #     img_array = np.array(image.convert('RGB')) # Work with RGB
# # # # #     num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
# # # # #     num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)

# # # # #     # Salt noise
# # # # #     coords_salt = [np.random.randint(0, i - 1 if i > 1 else 1, num_salt) for i in img_array.shape[:2]]
# # # # #     if num_salt > 0 : img_array[coords_salt[0], coords_salt[1], :] = [255,255,255]


# # # # #     # Pepper noise
# # # # #     coords_pepper = [np.random.randint(0, i - 1 if i > 1 else 1, num_pepper) for i in img_array.shape[:2]]
# # # # #     if num_pepper > 0: img_array[coords_pepper[0], coords_pepper[1], :] = [0,0,0]

# # # # #     return Image.fromarray(img_array)


# # # # # def add_speckle_noise(image):
# # # # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # # # #     noise = np.random.randn(*img_array.shape)
# # # # #     img_noisy = img_array + img_array * noise
# # # # #     img_noisy = np.clip(img_noisy, 0, 1)
# # # # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # # # #     return Image.fromarray(img_noisy)

# # # # # def add_motion_blur(image, size=9):
# # # # #     size = int(size)
# # # # #     if size <= 0 : return image
# # # # #     kernel = np.zeros((size, size))
# # # # #     kernel[int((size - 1)/2), :] = np.ones(size)
# # # # #     kernel = kernel / size
# # # # #     img_array = np.array(image.convert('RGB'))
# # # # #     img_blur = cv2.filter2D(img_array, -1, kernel)
# # # # #     return Image.fromarray(img_blur)

# # # # # # Occlusion Transformations
# # # # # def apply_cutout(image, mask_size):
# # # # #     mask_size = int(mask_size)
# # # # #     if mask_size <=0: return image
# # # # #     img_array = np.array(image.convert('RGB'))
# # # # #     h, w = img_array.shape[:2]
# # # # #     if h == 0 or w == 0: return image # Empty image

# # # # #     y = np.random.randint(h)
# # # # #     x = np.random.randint(w)
# # # # #     y1 = np.clip(y - mask_size // 2, 0, h)
# # # # #     y2 = np.clip(y + mask_size // 2, 0, h)
# # # # #     x1 = np.clip(x - mask_size // 2, 0, w)
# # # # #     x2 = np.clip(x + mask_size // 2, 0, w)
# # # # #     img_array[y1:y2, x1:x2] = 0  # Black box
# # # # #     return Image.fromarray(img_array)

# # # # # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
# # # # #     img_array = np.array(image.convert('RGB'))
# # # # #     h, w, c = img_array.shape # expect 3 channels for random color fill

# # # # #     s_img = h * w
# # # # #     s_erase = np.random.uniform(sl, sh) * s_img
# # # # #     r_aspect = np.random.uniform(r1, 1/r1)

# # # # #     h_e = int(np.sqrt(s_erase * r_aspect))
# # # # #     w_e = int(np.sqrt(s_erase / r_aspect))

# # # # #     if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h: # check if erase dimensions are valid
# # # # #         return Image.fromarray(img_array)

# # # # #     x_e = np.random.randint(0, w - w_e + 1) # +1 for upper bound
# # # # #     y_e = np.random.randint(0, h - h_e + 1)

# # # # #     img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
# # # # #     return Image.fromarray(img_array)

# # # # # # Mix Transformations
# # # # # def apply_mixup(image, other_image, alpha=0.4):
# # # # #     if other_image is None: return image
# # # # #     lam = np.random.beta(alpha, alpha)
# # # # #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# # # # #     # Ensure other_image is RGB and same size
# # # # #     other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
# # # # #     mixed_array = lam * image_array + (1 - lam) * other_array
# # # # #     return Image.fromarray(mixed_array.astype(np.uint8))

# # # # # def apply_cutmix(image, other_image):
# # # # #     if other_image is None: return image
# # # # #     img_array = np.array(image.convert('RGB'))
# # # # #     # Ensure other_image is RGB and same size
# # # # #     other_array = np.array(other_image.convert('RGB').resize(image.size))
    
# # # # #     h, w, _ = img_array.shape
# # # # #     if h == 0 or w == 0: return image

# # # # #     lam = np.random.beta(1.0, 1.0) # typically 1.0, 1.0 for cutmix
# # # # #     cut_ratio = np.sqrt(1. - lam)
# # # # #     cut_w = int(w * cut_ratio)
# # # # #     cut_h = int(h * cut_ratio)

# # # # #     if cut_w == 0 or cut_h == 0: return image # no actual cut

# # # # #     # uniform
# # # # #     cx = np.random.randint(w)
# # # # #     cy = np.random.randint(h)

# # # # #     bbx1 = np.clip(cx - cut_w // 2, 0, w)
# # # # #     bby1 = np.clip(cy - cut_h // 2, 0, h)
# # # # #     bbx2 = np.clip(cx + cut_w // 2, 0, w)
# # # # #     bby2 = np.clip(cy + cut_h // 2, 0, h)
    
# # # # #     if bbx1 < bbx2 and bby1 < bby2: # Ensure valid box
# # # # #       img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
# # # # #     return Image.fromarray(img_array)


# # # # # # Master Augmentation Application Function
# # # # # def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
# # # # #     """Apply a series of augmentations based on selected techniques and parameters."""
# # # # #     # Geometric Transformations
# # # # #     if "rotate" in techniques: # Renamed from "rotation" for consistency
# # # # #         angle = float(params.get("rotation_angle", 90))
# # # # #         image = rotate_image(image, angle)
# # # # #     if "scale" in techniques: # Renamed from "scaling"
# # # # #         scale_factor = float(params.get("scaling_factor", 1.5))
# # # # #         image = scale_image(image, scale_factor)
# # # # #     if "translate" in techniques: # Renamed from "translation"
# # # # #         x_offset = int(params.get("translation_x", 0))
# # # # #         y_offset = int(params.get("translation_y", 0))
# # # # #         image = translate_image(image, x_offset, y_offset)
# # # # #     if "flip_horizontal" in techniques: # Renamed
# # # # #         image = flip_horizontal(image)
# # # # #     if "flip_vertical" in techniques: # Renamed
# # # # #         image = flip_vertical(image)
# # # # #     if "crop" in techniques: # Renamed
# # # # #         left = int(params.get("crop_left", 0))
# # # # #         top = int(params.get("crop_top", 0))
# # # # #         right = int(params.get("crop_right", image.width if image else 0)) # Default to image width
# # # # #         bottom = int(params.get("crop_bottom", image.height if image else 0)) # Default to image height
# # # # #         image = crop_image(image, left, top, right, bottom)
# # # # #     if "pad" in techniques: # Renamed
# # # # #         padding = int(params.get("padding_size", 0))
# # # # #         padding_color = params.get("padding_color", "#000000")
# # # # #         image = pad_image(image, padding, padding_color)

# # # # #     # Color Transformations
# # # # #     if "brightness" in techniques:
# # # # #         image = adjust_brightness(image, float(params.get("brightness_factor", 1.0)))
# # # # #     if "contrast" in techniques:
# # # # #         image = adjust_contrast(image, float(params.get("contrast_factor", 1.0)))
# # # # #     if "grayscale" in techniques:
# # # # #         image = convert_grayscale(image)
# # # # #     if "saturation" in techniques:
# # # # #         image = adjust_saturation(image, float(params.get("saturation_factor", 1.0)))

# # # # #     # Noise Transformations
# # # # #     if "gaussian_noise" in techniques:
# # # # #         var = float(params.get("gaussian_variance", 0.01))
# # # # #         image = add_gaussian_noise(image, var=var)
# # # # #     if "salt_pepper_noise" in techniques:
# # # # #         amount = float(params.get("sap_amount", 0.005)) # salt and pepper amount
# # # # #         image = add_salt_pepper_noise(image, amount=amount)
# # # # #     if "speckle_noise" in techniques:
# # # # #         image = add_speckle_noise(image)
# # # # #     if "motion_blur" in techniques:
# # # # #         size = int(params.get("motion_blur_size", 9))
# # # # #         image = add_motion_blur(image, size=size)

# # # # #     # Occlusion Transformations
# # # # #     if "cutout" in techniques:
# # # # #         size = int(params.get("cutout_size", 50))
# # # # #         image = apply_cutout(image, size)
# # # # #     if "random_erasing" in techniques:
# # # # #         image = apply_random_erasing(image) # Uses default params in function

# # # # #     # Mixup and Cutmix (requires another image from the uploaded set)
# # # # #     other_image_for_mix = None
# # # # #     if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset:
# # # # #         possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
# # # # #         if not possible_other_files and len(files_in_dataset) > 0:
# # # # #             possible_other_files = files_in_dataset # Use any file if current is the only one or not found

# # # # #         if possible_other_files:
# # # # #             other_filename = random.choice(possible_other_files)
# # # # #             other_filepath = os.path.join(source_dataset_folder, other_filename)
# # # # #             try:
# # # # #                 other_image_for_mix = Image.open(other_filepath)
# # # # #             except Exception as e:
# # # # #                 print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
# # # # #                 other_image_for_mix = None
# # # # #         else:
# # # # #             print("Warning: No other images available for mixup/cutmix.")

# # # # #     if "mixup" in techniques and other_image_for_mix:
# # # # #         alpha = float(params.get("mixup_alpha", 0.4))
# # # # #         image = apply_mixup(image, other_image_for_mix, alpha)
# # # # #     if "cutmix" in techniques and other_image_for_mix:
# # # # #         image = apply_cutmix(image, other_image_for_mix)
    
# # # # #     if other_image_for_mix: # Close the image if opened
# # # # #         other_image_for_mix.close()

# # # # #     return image

# # # # # # --- End of Augmentation Functions ---

# # # # # @app.route('/upload', methods=['POST'])
# # # # # def upload_files():
# # # # #     dataset = request.form.get('dataset')
# # # # #     files = request.files.getlist('files')

# # # # #     if not dataset:
# # # # #         return "No dataset name provided", 400

# # # # #     dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
# # # # #     os.makedirs(dataset_folder, exist_ok=True)

# # # # #     for file in files:
# # # # #         filename = secure_filename(file.filename)
# # # # #         filepath = os.path.join(dataset_folder, filename)
# # # # #         file.save(filepath)

# # # # #         if filename.lower().endswith('.zip'):
# # # # #             try:
# # # # #                 with zipfile.ZipFile(filepath, 'r') as zip_ref:
# # # # #                     zip_ref.extractall(dataset_folder)
# # # # #                 os.remove(filepath) # Remove zip after extraction
# # # # #             except zipfile.BadZipFile:
# # # # #                 print(f"Bad zip file: {filepath}") # Keep the bad zip for inspection
# # # # #                 # Or return an error to the user:
# # # # #                 # return f"Uploaded file {filename} is not a valid ZIP file.", 400


# # # # #     return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


# # # # # @app.route('/datasets', methods=['GET'])
# # # # # def list_datasets():
# # # # #     datasets_info = []
# # # # #     base_upload_folder = app.config['UPLOAD_FOLDER']
# # # # #     for folder_name in os.listdir(base_upload_folder):
# # # # #         folder_path = os.path.join(base_upload_folder, folder_name)
# # # # #         if os.path.isdir(folder_path):
# # # # #             files_in_folder = os.listdir(folder_path)
# # # # #             image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
# # # # #             metadata = augmentation_metadata.get(folder_name, {})
# # # # #             datasets_info.append({
# # # # #                 'name': folder_name,
# # # # #                 'count': len(image_files),
# # # # #                 'files': files_in_folder, # List all files, not just images, for completeness
# # # # #                 'techniques': metadata.get("techniques", []),
# # # # #                 'parameters': metadata.get("parameters", {}),
# # # # #                 'augmented_zip': metadata.get("augmented_zip")
# # # # #             })
# # # # #     return jsonify(datasets_info)


# # # # # @app.route('/augment', methods=['POST'])
# # # # # def augment_dataset_route():
# # # # #     data = request.get_json()
# # # # #     dataset_name = secure_filename(data.get('datasetName'))
# # # # #     techniques = data.get('techniques', [])
# # # # #     parameters = data.get('parameters', {})

# # # # #     if not dataset_name:
# # # # #         return jsonify({'error': 'Dataset name not provided'}), 400
# # # # #     if not techniques:
# # # # #         return jsonify({'error': 'No augmentation techniques selected'}), 400


# # # # #     source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
# # # # #     target_folder_base = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
    
# # # # #     # Create a unique subfolder for this augmentation run to avoid overwriting
# # # # #     run_index = 0
# # # # #     target_folder = os.path.join(target_folder_base, f"run_{run_index}")
# # # # #     while os.path.exists(target_folder):
# # # # #         run_index += 1
# # # # #         target_folder = os.path.join(target_folder_base, f"run_{run_index}")
# # # # #     os.makedirs(target_folder, exist_ok=True)


# # # # #     if not os.path.isdir(source_folder):
# # # # #         return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

# # # # #     source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # # # #     augmented_file_names = []
# # # # #     for filename in source_image_files:
# # # # #         img_path = os.path.join(source_folder, filename)
# # # # #         try:
# # # # #             with Image.open(img_path) as img:
# # # # #                 # Pass necessary info for mixup/cutmix
# # # # #                 augmented_img = run_augmentations(
# # # # #                     img.copy(), # Pass a copy to avoid issues with original image object
# # # # #                     techniques,
# # # # #                     parameters,
# # # # #                     source_folder,          # For Mixup/Cutmix to load other images
# # # # #                     source_image_files,     # List of image names in source_folder
# # # # #                     filename                # Current image filename
# # # # #                 )
                
# # # # #                 # Ensure mode is suitable for saving (e.g., convert P mode if it has alpha)
# # # # #                 if augmented_img.mode == 'P' and 'transparency' in augmented_img.info:
# # # # #                     augmented_img = augmented_img.convert("RGBA")
# # # # #                 elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): # Grayscale with alpha
# # # # #                     augmented_img = augmented_img.convert("RGBA") # or handle alpha appropriately
# # # # #                 elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: # L for grayscale
# # # # #                      augmented_img = augmented_img.convert('RGB')


# # # # #                 # Determine save format based on original or default to PNG for augmentations
# # # # #                 base, ext = os.path.splitext(filename)
# # # # #                 save_filename = f"aug_{base}{ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'}"
# # # # #                 save_path = os.path.join(target_folder, save_filename)
                
# # # # #                 if ext.lower() in ['.jpg', '.jpeg']:
# # # # #                     augmented_img.save(save_path, "JPEG", quality=95)
# # # # #                 else: # Default to PNG
# # # # #                     augmented_img.save(save_path, "PNG")
# # # # #                 augmented_file_names.append(save_filename)

# # # # #         except Exception as e:
# # # # #             print(f"Error augmenting image {filename}: {e}")
# # # # #             # Optionally, copy original if augmentation fails
# # # # #             # import shutil
# # # # #             # shutil.copy(img_path, os.path.join(target_folder, f"err_orig_{filename}"))


# # # # #     # Zip the augmented folder (only the current run)
# # # # #     zip_filename = f"{dataset_name}_augmented_run_{run_index}.zip"
# # # # #     # Save zip inside the specific run's folder or one level up in AUGMENTED_FOLDER/dataset_name
# # # # #     zip_path = os.path.join(target_folder_base, zip_filename) # Store zip one level up from run_x folder
    
# # # # #     with zipfile.ZipFile(zip_path, 'w') as zipf:
# # # # #         for aug_file in augmented_file_names: # os.listdir(target_folder)
# # # # #             file_to_zip_path = os.path.join(target_folder, aug_file)
# # # # #             if os.path.isfile(file_to_zip_path) and aug_file.lower().endswith(('.png', '.jpg', '.jpeg')):
# # # # #                  zipf.write(file_to_zip_path, aug_file)



# # # # #     # Save metadata (associating with the base dataset name)
# # # # #     # This could be extended to store a list of augmentation runs if needed
# # # # #     augmentation_metadata[dataset_name] = {
# # # # #         "techniques": techniques,
# # # # #         "parameters": parameters,
# # # # #         "augmented_zip": zip_filename, # This will point to the latest zip
# # # # #         "last_augmented_run_folder": target_folder # Store the path to the actual image files
# # # # #     }
# # # # #     with open(METADATA_FILE, 'w') as f:
# # # # #         json.dump(augmentation_metadata, f, indent=4)

# # # # #     return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'augmented_files_path': target_folder })


# # # # # @app.route('/uploads/<dataset>/<filename>')
# # # # # def uploaded_file(dataset, filename):
# # # # #     return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

# # # # # @app.route('/augmented/<dataset>/<zipfilename>') # This serves the ZIP file
# # # # # def serve_augmented_zip(dataset, zipfilename):
# # # # #     # Zips are stored in AUGMENTED_FOLDER/dataset_name/zipfilename
# # # # #     return send_from_directory(os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset)), secure_filename(zipfilename), as_attachment=True)

# # # # # # To view individual augmented images if needed (requires knowing the run folder)
# # # # # # This endpoint might need adjustment if you want to browse specific runs.
# # # # # # For simplicity, the ZIP download is primary.
# # # # # # If you want to show augmented images in UI before zipping, that's a different flow.
# # # # # @app.route('/augmented_image/<dataset>/<run_folder>/<filename>')
# # # # # def serve_augmented_image(dataset, run_folder, filename):
# # # # #     # Path: augmented/dataset_name/run_folder/filename
# # # # #     return send_from_directory(os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_folder)), secure_filename(filename))


# # # # # if __name__ == '__main__':
# # # # #     app.run(host='0.0.0.0', port=5001, debug=True)




# # # # import os
# # # # import zipfile
# # # # import json
# # # # import random
# # # # import numpy as np
# # # # import cv2 # For motion blur
# # # # import datetime # For timestamps

# # # # from flask import Flask, request, jsonify, send_from_directory, current_app
# # # # from flask_cors import CORS
# # # # from werkzeug.utils import secure_filename
# # # # from PIL import Image, ImageOps, ImageEnhance, ImageChops

# # # # app = Flask(__name__)
# # # # CORS(app)

# # # # UPLOAD_FOLDER = 'uploads'
# # # # AUGMENTED_FOLDER = 'augmented'
# # # # METADATA_FILE = 'augmentation_metadata.json'

# # # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # # app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # # # # Load augmentation metadata
# # # # if os.path.exists(METADATA_FILE):
# # # #     with open(METADATA_FILE, 'r') as f:
# # # #         augmentation_metadata = json.load(f)
# # # # else:
# # # #     augmentation_metadata = {}

# # # # # --- Start of Augmentation Functions (Identical to previous version) ---

# # # # # Geometric Transformations
# # # # def rotate_image(image, angle=90):
# # # #     return image.rotate(angle, expand=True)

# # # # def scale_image(image, scale=1.5):
# # # #     if scale <= 0: return image # Avoid invalid scale
# # # #     width, height = image.size
# # # #     return image.resize((int(width * scale), int(height * scale)))

# # # # def translate_image(image, x_offset, y_offset):
# # # #     return ImageChops.offset(image, x_offset, y_offset)

# # # # def flip_horizontal(image):
# # # #     return ImageOps.mirror(image)

# # # # def flip_vertical(image):
# # # #     return ImageOps.flip(image)

# # # # def crop_image(image, left, top, right, bottom):
# # # #     width, height = image.size
# # # #     left = max(0, int(left))
# # # #     top = max(0, int(top))
# # # #     right = min(width, int(right))
# # # #     bottom = min(height, int(bottom))
# # # #     if left < right and top < bottom:
# # # #         return image.crop((left, top, right, bottom))
# # # #     return image

# # # # def pad_image(image, padding_size, padding_color="#000000"):
# # # #     padding_size = int(padding_size)
# # # #     if padding_size <=0: return image
# # # #     return ImageOps.expand(image, border=padding_size, fill=padding_color)

# # # # # Color Transformations
# # # # def adjust_brightness(image, factor):
# # # #     enhancer = ImageEnhance.Brightness(image.convert('RGB'))
# # # #     return enhancer.enhance(factor)

# # # # def adjust_contrast(image, factor):
# # # #     enhancer = ImageEnhance.Contrast(image.convert('RGB'))
# # # #     return enhancer.enhance(factor)

# # # # def convert_grayscale(image):
# # # #     return ImageOps.grayscale(image).convert('RGB')

# # # # def adjust_saturation(image, factor):
# # # #     enhancer = ImageEnhance.Color(image.convert('RGB'))
# # # #     return enhancer.enhance(factor)

# # # # # Noise Transformations
# # # # def add_gaussian_noise(image, mean=0, var=0.01):
# # # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # # #     noise = np.random.normal(mean, var ** 0.5, img_array.shape)
# # # #     img_noisy = img_array + noise
# # # #     img_noisy = np.clip(img_noisy, 0, 1)
# # # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # # #     return Image.fromarray(img_noisy)

# # # # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# # # #     img_array = np.array(image.convert('RGB'))
# # # #     num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
# # # #     num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)
# # # #     coords_salt = [np.random.randint(0, i - 1 if i > 1 else 1, num_salt) for i in img_array.shape[:2]]
# # # #     if num_salt > 0 : img_array[coords_salt[0], coords_salt[1], :] = [255,255,255]
# # # #     coords_pepper = [np.random.randint(0, i - 1 if i > 1 else 1, num_pepper) for i in img_array.shape[:2]]
# # # #     if num_pepper > 0: img_array[coords_pepper[0], coords_pepper[1], :] = [0,0,0]
# # # #     return Image.fromarray(img_array)

# # # # def add_speckle_noise(image):
# # # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # # #     noise = np.random.randn(*img_array.shape)
# # # #     img_noisy = img_array + img_array * noise
# # # #     img_noisy = np.clip(img_noisy, 0, 1)
# # # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # # #     return Image.fromarray(img_noisy)

# # # # def add_motion_blur(image, size=9):
# # # #     size = int(size)
# # # #     if size <= 0 : return image
# # # #     kernel = np.zeros((size, size))
# # # #     kernel[int((size - 1)/2), :] = np.ones(size)
# # # #     kernel = kernel / size
# # # #     img_array = np.array(image.convert('RGB'))
# # # #     img_blur = cv2.filter2D(img_array, -1, kernel)
# # # #     return Image.fromarray(img_blur)

# # # # # Occlusion Transformations
# # # # def apply_cutout(image, mask_size):
# # # #     mask_size = int(mask_size)
# # # #     if mask_size <=0: return image
# # # #     img_array = np.array(image.convert('RGB'))
# # # #     h, w = img_array.shape[:2]
# # # #     if h == 0 or w == 0: return image
# # # #     y = np.random.randint(h)
# # # #     x = np.random.randint(w)
# # # #     y1 = np.clip(y - mask_size // 2, 0, h)
# # # #     y2 = np.clip(y + mask_size // 2, 0, h)
# # # #     x1 = np.clip(x - mask_size // 2, 0, w)
# # # #     x2 = np.clip(x + mask_size // 2, 0, w)
# # # #     img_array[y1:y2, x1:x2] = 0
# # # #     return Image.fromarray(img_array)

# # # # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
# # # #     img_array = np.array(image.convert('RGB'))
# # # #     h, w, c = img_array.shape
# # # #     s_img = h * w
# # # #     s_erase = np.random.uniform(sl, sh) * s_img
# # # #     r_aspect = np.random.uniform(r1, 1/r1)
# # # #     h_e = int(np.sqrt(s_erase * r_aspect))
# # # #     w_e = int(np.sqrt(s_erase / r_aspect))
# # # #     if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h:
# # # #         return Image.fromarray(img_array)
# # # #     x_e = np.random.randint(0, w - w_e + 1)
# # # #     y_e = np.random.randint(0, h - h_e + 1)
# # # #     img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
# # # #     return Image.fromarray(img_array)

# # # # # Mix Transformations
# # # # def apply_mixup(image, other_image, alpha=0.4):
# # # #     if other_image is None: return image
# # # #     lam = np.random.beta(alpha, alpha)
# # # #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# # # #     other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
# # # #     mixed_array = lam * image_array + (1 - lam) * other_array
# # # #     return Image.fromarray(mixed_array.astype(np.uint8))

# # # # def apply_cutmix(image, other_image):
# # # #     if other_image is None: return image
# # # #     img_array = np.array(image.convert('RGB'))
# # # #     other_array = np.array(other_image.convert('RGB').resize(image.size))
# # # #     h, w, _ = img_array.shape
# # # #     if h == 0 or w == 0: return image
# # # #     lam = np.random.beta(1.0, 1.0)
# # # #     cut_ratio = np.sqrt(1. - lam)
# # # #     cut_w = int(w * cut_ratio)
# # # #     cut_h = int(h * cut_ratio)
# # # #     if cut_w == 0 or cut_h == 0: return image
# # # #     cx = np.random.randint(w)
# # # #     cy = np.random.randint(h)
# # # #     bbx1 = np.clip(cx - cut_w // 2, 0, w)
# # # #     bby1 = np.clip(cy - cut_h // 2, 0, h)
# # # #     bbx2 = np.clip(cx + cut_w // 2, 0, w)
# # # #     bby2 = np.clip(cy + cut_h // 2, 0, h)
# # # #     if bbx1 < bbx2 and bby1 < bby2:
# # # #       img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
# # # #     return Image.fromarray(img_array)

# # # # # Master Augmentation Application Function
# # # # def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
# # # #     # Geometric Transformations
# # # #     if "rotate" in techniques:
# # # #         angle = float(params.get("rotation_angle", 90))
# # # #         image = rotate_image(image, angle)
# # # #     if "scale" in techniques:
# # # #         scale_factor = float(params.get("scaling_factor", 1.5))
# # # #         image = scale_image(image, scale_factor)
# # # #     if "translate" in techniques:
# # # #         x_offset = int(params.get("translation_x", 0))
# # # #         y_offset = int(params.get("translation_y", 0))
# # # #         image = translate_image(image, x_offset, y_offset)
# # # #     if "flip_horizontal" in techniques:
# # # #         image = flip_horizontal(image)
# # # #     if "flip_vertical" in techniques:
# # # #         image = flip_vertical(image)
# # # #     if "crop" in techniques:
# # # #         left = int(params.get("crop_left", 0))
# # # #         top = int(params.get("crop_top", 0))
# # # #         right = int(params.get("crop_right", image.width if image else 0))
# # # #         bottom = int(params.get("crop_bottom", image.height if image else 0))
# # # #         image = crop_image(image, left, top, right, bottom)
# # # #     if "pad" in techniques:
# # # #         padding = int(params.get("padding_size", 0))
# # # #         padding_color = params.get("padding_color", "#000000")
# # # #         image = pad_image(image, padding, padding_color)

# # # #     # Color Transformations
# # # #     if "brightness" in techniques:
# # # #         image = adjust_brightness(image, float(params.get("brightness_factor", 1.0)))
# # # #     if "contrast" in techniques:
# # # #         image = adjust_contrast(image, float(params.get("contrast_factor", 1.0)))
# # # #     if "grayscale" in techniques:
# # # #         image = convert_grayscale(image)
# # # #     if "saturation" in techniques:
# # # #         image = adjust_saturation(image, float(params.get("saturation_factor", 1.0)))

# # # #     # Noise Transformations
# # # #     if "gaussian_noise" in techniques:
# # # #         var = float(params.get("gaussian_variance", 0.01))
# # # #         image = add_gaussian_noise(image, var=var)
# # # #     if "salt_pepper_noise" in techniques:
# # # #         amount = float(params.get("sap_amount", 0.005))
# # # #         image = add_salt_pepper_noise(image, amount=amount)
# # # #     if "speckle_noise" in techniques:
# # # #         image = add_speckle_noise(image)
# # # #     if "motion_blur" in techniques:
# # # #         size = int(params.get("motion_blur_size", 9))
# # # #         image = add_motion_blur(image, size=size)

# # # #     # Occlusion Transformations
# # # #     if "cutout" in techniques:
# # # #         size = int(params.get("cutout_size", 50))
# # # #         image = apply_cutout(image, size)
# # # #     if "random_erasing" in techniques:
# # # #         image = apply_random_erasing(image)

# # # #     # Mixup and Cutmix
# # # #     other_image_for_mix = None
# # # #     if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset:
# # # #         possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
# # # #         if not possible_other_files and len(files_in_dataset) > 0:
# # # #             possible_other_files = files_in_dataset
# # # #         if possible_other_files:
# # # #             other_filename = random.choice(possible_other_files)
# # # #             other_filepath = os.path.join(source_dataset_folder, other_filename)
# # # #             try:
# # # #                 other_image_for_mix = Image.open(other_filepath)
# # # #             except Exception as e:
# # # #                 print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
# # # #                 other_image_for_mix = None
# # # #         else:
# # # #             print("Warning: No other images available for mixup/cutmix.")

# # # #     if "mixup" in techniques and other_image_for_mix:
# # # #         alpha = float(params.get("mixup_alpha", 0.4))
# # # #         image = apply_mixup(image, other_image_for_mix, alpha)
# # # #     if "cutmix" in techniques and other_image_for_mix:
# # # #         image = apply_cutmix(image, other_image_for_mix)
    
# # # #     if other_image_for_mix:
# # # #         other_image_for_mix.close()
# # # #     return image

# # # # # --- End of Augmentation Functions ---

# # # # @app.route('/upload', methods=['POST'])
# # # # def upload_files():
# # # #     dataset = request.form.get('dataset')
# # # #     files = request.files.getlist('files')

# # # #     if not dataset:
# # # #         return "No dataset name provided", 400

# # # #     dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
# # # #     os.makedirs(dataset_folder, exist_ok=True)

# # # #     for file in files:
# # # #         filename = secure_filename(file.filename)
# # # #         filepath = os.path.join(dataset_folder, filename)
# # # #         file.save(filepath)

# # # #         if filename.lower().endswith('.zip'):
# # # #             try:
# # # #                 with zipfile.ZipFile(filepath, 'r') as zip_ref:
# # # #                     zip_ref.extractall(dataset_folder)
# # # #                 os.remove(filepath)
# # # #             except zipfile.BadZipFile:
# # # #                 print(f"Bad zip file: {filepath}")
# # # #     return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


# # # # @app.route('/datasets', methods=['GET'])
# # # # def list_datasets():
# # # #     datasets_info = []
# # # #     base_upload_folder = app.config['UPLOAD_FOLDER']
# # # #     for folder_name in os.listdir(base_upload_folder):
# # # #         folder_path = os.path.join(base_upload_folder, folder_name)
# # # #         if os.path.isdir(folder_path):
# # # #             files_in_folder = os.listdir(folder_path)
# # # #             image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
# # # #             dataset_metadata = augmentation_metadata.get(folder_name, {})
# # # #             # augmentation_runs should be a list, defaulting to empty if not present
# # # #             runs = dataset_metadata.get("augmentation_runs", []) 
            
# # # #             datasets_info.append({
# # # #                 'name': folder_name,
# # # #                 'count': len(image_files),
# # # #                 'files': files_in_folder,
# # # #                 'augmentation_runs': runs # Pass the whole list of runs
# # # #             })
# # # #     return jsonify(datasets_info)


# # # # @app.route('/augment', methods=['POST'])
# # # # def augment_dataset_route():
# # # #     data = request.get_json()
# # # #     dataset_name = secure_filename(data.get('datasetName'))
# # # #     techniques = data.get('techniques', [])
# # # #     parameters = data.get('parameters', {})

# # # #     if not dataset_name:
# # # #         return jsonify({'error': 'Dataset name not provided'}), 400
# # # #     if not techniques:
# # # #         return jsonify({'error': 'No augmentation techniques selected'}), 400

# # # #     source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
# # # #     if not os.path.isdir(source_folder):
# # # #         return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

# # # #     # --- Create unique run folder and ZIP name ---
# # # #     dataset_augmented_base_path = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
# # # #     os.makedirs(dataset_augmented_base_path, exist_ok=True)

# # # #     run_index = 0
# # # #     current_run_id = ""
# # # #     target_run_folder_path = "" # Path to augmented/<dataset_name>/run_X
# # # #     while True:
# # # #         current_run_id = f"run_{run_index}"
# # # #         target_run_folder_path = os.path.join(dataset_augmented_base_path, current_run_id)
# # # #         if not os.path.exists(target_run_folder_path):
# # # #             os.makedirs(target_run_folder_path)
# # # #             break
# # # #         run_index += 1
    
# # # #     zip_filename = f"{dataset_name}_augmented_{current_run_id}.zip"
# # # #     # ZIP file will be stored in augmented/<dataset_name>/<zip_filename>
# # # #     zip_filepath = os.path.join(dataset_augmented_base_path, zip_filename)
# # # #     # --- End of unique folder/ZIP name creation ---

# # # #     source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# # # #     augmented_file_names_for_zip = []

# # # #     for filename in source_image_files:
# # # #         img_path = os.path.join(source_folder, filename)
# # # #         try:
# # # #             with Image.open(img_path) as img:
# # # #                 augmented_img = run_augmentations(
# # # #                     img.copy(), techniques, parameters,
# # # #                     source_folder, source_image_files, filename
# # # #                 )
                
# # # #                 if augmented_img.mode == 'P' and 'transparency' in augmented_img.info:
# # # #                     augmented_img = augmented_img.convert("RGBA")
# # # #                 elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info):
# # # #                     augmented_img = augmented_img.convert("RGBA")
# # # #                 elif augmented_img.mode not in ['RGB', 'RGBA', 'L']:
# # # #                      augmented_img = augmented_img.convert('RGB')

# # # #                 base, ext = os.path.splitext(filename)
# # # #                 save_filename = f"aug_{base}{ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'}"
# # # #                 # Save augmented image into the specific run folder
# # # #                 save_path = os.path.join(target_run_folder_path, save_filename)
                
# # # #                 if ext.lower() in ['.jpg', '.jpeg']:
# # # #                     augmented_img.save(save_path, "JPEG", quality=95)
# # # #                 else:
# # # #                     augmented_img.save(save_path, "PNG")
# # # #                 augmented_file_names_for_zip.append(save_filename)
# # # #         except Exception as e:
# # # #             print(f"Error augmenting image {filename}: {e}")

# # # #     # Zip the contents of the specific run's folder
# # # #     with zipfile.ZipFile(zip_filepath, 'w') as zipf:
# # # #         for aug_file_name in augmented_file_names_for_zip:
# # # #             # Path to the file inside the run folder
# # # #             file_in_run_folder_path = os.path.join(target_run_folder_path, aug_file_name)
# # # #             if os.path.isfile(file_in_run_folder_path):
# # # #                  zipf.write(file_in_run_folder_path, aug_file_name) # Add to zip with its name, not full path

# # # #     # --- Update Metadata ---
# # # #     if dataset_name not in augmentation_metadata:
# # # #         augmentation_metadata[dataset_name] = {"augmentation_runs": []}
# # # #     elif "augmentation_runs" not in augmentation_metadata[dataset_name]: # Ensure list exists
# # # #         augmentation_metadata[dataset_name]["augmentation_runs"] = []

# # # #     new_run_info = {
# # # #         "run_id": current_run_id,
# # # #         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
# # # #         "techniques": techniques,
# # # #         "parameters": parameters,
# # # #         "augmented_zip": zip_filename, # Just the filename of the zip
# # # #         "output_folder_name": current_run_id # Name of the subfolder like "run_0"
# # # #     }
# # # #     augmentation_metadata[dataset_name]["augmentation_runs"].append(new_run_info)
    
# # # #     with open(METADATA_FILE, 'w') as f:
# # # #         json.dump(augmentation_metadata, f, indent=4)

# # # #     return jsonify({
# # # #         'message': 'Augmentation complete', 
# # # #         'zip_filename': zip_filename, 
# # # #         'run_id': current_run_id,
# # # #         'augmented_images_path': target_run_folder_path # Path to the folder with this run's images
# # # #     })



# # # # @app.route('/uploads/<dataset>/<filename>')
# # # # def uploaded_file(dataset, filename):
# # # #     return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

# # # # # Serves the ZIP file for a given dataset and zipfilename
# # # # @app.route('/augmented/<dataset>/<zipfilename>') 
# # # # def serve_augmented_zip(dataset, zipfilename):
# # # #     # Zips are stored directly under AUGMENTED_FOLDER/dataset_name/
# # # #     dataset_aug_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset))
# # # #     return send_from_directory(dataset_aug_path, secure_filename(zipfilename), as_attachment=True)

# # # # # To view individual augmented images from a specific run (if needed by UI in future)
# # # # @app.route('/augmented_image/<dataset>/<run_id>/<filename>')
# # # # def serve_augmented_image(dataset, run_id, filename):
# # # #     # Path: augmented/dataset_name/run_id/filename
# # # #     image_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_id))
# # # #     return send_from_directory(image_path, secure_filename(filename))


# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5001, debug=True)




# # # >>>>>>> Stashed changes
# # # import os
# # # import zipfile
# # # import json
# # # import random
# # # import numpy as np
# # # import cv2 # For motion blur
# # # import datetime # For timestamps
# # # import io # For image byte streaming
# # # import base64 # For base64 encoding

# # # from flask import Flask, request, jsonify, send_from_directory
# # # from flask_cors import CORS
# # # from werkzeug.utils import secure_filename
# # # from PIL import Image, ImageOps, ImageEnhance, ImageChops

# # # app = Flask(__name__)
# # # CORS(app)

# # # UPLOAD_FOLDER = 'uploads'
# # # AUGMENTED_FOLDER = 'augmented'
# # # METADATA_FILE = 'augmentation_metadata.json'

# # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # # if os.path.exists(METADATA_FILE):
# # #     with open(METADATA_FILE, 'r') as f:
# # #         augmentation_metadata = json.load(f)
# # # else:
# # #     augmentation_metadata = {}

# # # # --- Start of Augmentation Functions (Identical to previous full version) ---
# # # # Geometric Transformations
# # # def rotate_image(image, angle=90):
# # #     return image.rotate(angle, expand=True)

# # # def scale_image(image, scale=1.5):
# # #     if scale <= 0: return image 
# # #     width, height = image.size
# # #     return image.resize((int(width * scale), int(height * scale)))

# # # def translate_image(image, x_offset, y_offset):
# # #     return ImageChops.offset(image, x_offset, y_offset)

# # # def flip_horizontal(image):
# # #     return ImageOps.mirror(image)

# # # def flip_vertical(image):
# # #     return ImageOps.flip(image)

# # # def crop_image(image, left, top, right, bottom):
# # #     width, height = image.size
# # #     left = max(0, int(left))
# # #     top = max(0, int(top))
# # #     right = min(width, int(right))
# # #     bottom = min(height, int(bottom))
# # #     if left < right and top < bottom:
# # #         return image.crop((left, top, right, bottom))
# # #     return image

# # # def pad_image(image, padding_size, padding_color="#000000"):
# # #     padding_size = int(padding_size)
# # #     if padding_size <=0: return image
# # #     return ImageOps.expand(image, border=padding_size, fill=padding_color)

# # # # Color Transformations
# # # def adjust_brightness(image, factor):
# # #     enhancer = ImageEnhance.Brightness(image.convert('RGB'))
# # #     return enhancer.enhance(factor)

# # # def adjust_contrast(image, factor):
# # #     enhancer = ImageEnhance.Contrast(image.convert('RGB'))
# # #     return enhancer.enhance(factor)

# # # def convert_grayscale(image):
# # #     return ImageOps.grayscale(image).convert('RGB')

# # # def adjust_saturation(image, factor):
# # #     enhancer = ImageEnhance.Color(image.convert('RGB'))
# # #     return enhancer.enhance(factor)

# # # # Noise Transformations
# # # def add_gaussian_noise(image, mean=0, var=0.01):
# # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # #     noise = np.random.normal(mean, var ** 0.5, img_array.shape)
# # #     img_noisy = img_array + noise
# # #     img_noisy = np.clip(img_noisy, 0, 1)
# # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # #     return Image.fromarray(img_noisy)

# # # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# # #     img_array = np.array(image.convert('RGB'))
# # #     num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
# # #     num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)
# # #     if img_array.size > 0 : # Check if image is not empty
# # #         coords_salt = [np.random.randint(0, i - 1 if i > 1 else 1, num_salt) for i in img_array.shape[:2]]
# # #         if num_salt > 0 : img_array[coords_salt[0], coords_salt[1], :] = [255,255,255]
# # #         coords_pepper = [np.random.randint(0, i - 1 if i > 1 else 1, num_pepper) for i in img_array.shape[:2]]
# # #         if num_pepper > 0: img_array[coords_pepper[0], coords_pepper[1], :] = [0,0,0]
# # #     return Image.fromarray(img_array)

# # # def add_speckle_noise(image):
# # #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# # #     noise = np.random.randn(*img_array.shape)
# # #     img_noisy = img_array + img_array * noise
# # #     img_noisy = np.clip(img_noisy, 0, 1)
# # #     img_noisy = (img_noisy * 255).astype(np.uint8)
# # #     return Image.fromarray(img_noisy)

# # # def add_motion_blur(image, size=9):
# # #     size = int(size)
# # #     if size <= 1 or size % 2 == 0 : size = 3 # Ensure odd kernel size > 1
# # #     kernel = np.zeros((size, size))
# # #     kernel[int((size - 1)/2), :] = np.ones(size)
# # #     kernel = kernel / size
# # #     img_array = np.array(image.convert('RGB'))
# # #     img_blur = cv2.filter2D(img_array, -1, kernel)
# # #     return Image.fromarray(img_blur)

# # # # Occlusion Transformations
# # # def apply_cutout(image, mask_size):
# # #     mask_size = int(mask_size)
# # #     if mask_size <=0: return image
# # #     img_array = np.array(image.convert('RGB'))
# # #     h, w = img_array.shape[:2]
# # #     if h == 0 or w == 0: return image
# # #     y = np.random.randint(h)
# # #     x = np.random.randint(w)
# # #     y1 = np.clip(y - mask_size // 2, 0, h)
# # #     y2 = np.clip(y + mask_size // 2, 0, h)
# # #     x1 = np.clip(x - mask_size // 2, 0, w)
# # #     x2 = np.clip(x + mask_size // 2, 0, w)
# # #     img_array[y1:y2, x1:x2] = 0
# # #     return Image.fromarray(img_array)

# # # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
# # #     img_array = np.array(image.convert('RGB'))
# # #     h, w, c = img_array.shape
# # #     s_img = h * w
# # #     s_erase = np.random.uniform(sl, sh) * s_img
# # #     r_aspect = np.random.uniform(r1, 1/r1)
# # #     h_e = int(np.sqrt(s_erase * r_aspect))
# # #     w_e = int(np.sqrt(s_erase / r_aspect))
# # #     if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h:
# # #         return Image.fromarray(img_array)
# # #     x_e = np.random.randint(0, w - w_e + 1)
# # #     y_e = np.random.randint(0, h - h_e + 1)
# # #     img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
# # #     return Image.fromarray(img_array)

# # # # Mix Transformations
# # # def apply_mixup(image, other_image, alpha=0.4):
# # #     if other_image is None: return image
# # #     lam = np.random.beta(alpha, alpha)
# # #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# # #     other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
# # #     mixed_array = lam * image_array + (1 - lam) * other_array
# # #     return Image.fromarray(mixed_array.astype(np.uint8))

# # # def apply_cutmix(image, other_image):
# # #     if other_image is None: return image
# # #     img_array = np.array(image.convert('RGB'))
# # #     other_array = np.array(other_image.convert('RGB').resize(image.size))
# # #     h, w, _ = img_array.shape
# # #     if h == 0 or w == 0: return image
# # #     lam = np.random.beta(1.0, 1.0)
# # #     cut_ratio = np.sqrt(1. - lam)
# # #     cut_w = int(w * cut_ratio)
# # #     cut_h = int(h * cut_ratio)
# # #     if cut_w == 0 or cut_h == 0: return image
# # #     cx = np.random.randint(w)
# # #     cy = np.random.randint(h)
# # #     bbx1 = np.clip(cx - cut_w // 2, 0, w)
# # #     bby1 = np.clip(cy - cut_h // 2, 0, h)
# # #     bbx2 = np.clip(cx + cut_w // 2, 0, w)
# # #     bby2 = np.clip(cy + cut_h // 2, 0, h)
# # #     if bbx1 < bbx2 and bby1 < bby2:
# # #       img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
# # #     return Image.fromarray(img_array)

# # # # Master Augmentation Application Function
# # # def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
# # #     processed_image = image.copy() # Work on a copy
# # #     # Geometric Transformations
# # #     if "rotate" in techniques:
# # #         angle = float(params.get("rotation_angle", 90))
# # #         processed_image = rotate_image(processed_image, angle)
# # #     if "scale" in techniques:
# # #         scale_factor = float(params.get("scaling_factor", 1.5))
# # #         processed_image = scale_image(processed_image, scale_factor)
# # #     # ... (all other transformations from previous version) ...
# # #     if "translate" in techniques:
# # #         x_offset = int(params.get("translation_x", 0))
# # #         y_offset = int(params.get("translation_y", 0))
# # #         processed_image = translate_image(processed_image, x_offset, y_offset)
# # #     if "flip_horizontal" in techniques:
# # #         processed_image = flip_horizontal(processed_image)
# # #     if "flip_vertical" in techniques:
# # #         processed_image = flip_vertical(processed_image)
# # #     if "crop" in techniques:
# # #         left = int(params.get("crop_left", 0))
# # #         top = int(params.get("crop_top", 0))
# # #         right = int(params.get("crop_right", processed_image.width if processed_image else 0))
# # #         bottom = int(params.get("crop_bottom", processed_image.height if processed_image else 0))
# # #         processed_image = crop_image(processed_image, left, top, right, bottom)
# # #     if "pad" in techniques:
# # #         padding = int(params.get("padding_size", 0))
# # #         padding_color = params.get("padding_color", "#000000")
# # #         processed_image = pad_image(processed_image, padding, padding_color)

# # #     # Color Transformations
# # #     if "brightness" in techniques:
# # #         processed_image = adjust_brightness(processed_image, float(params.get("brightness_factor", 1.0)))
# # #     if "contrast" in techniques:
# # #         processed_image = adjust_contrast(processed_image, float(params.get("contrast_factor", 1.0)))
# # #     if "grayscale" in techniques:
# # #         processed_image = convert_grayscale(processed_image)
# # #     if "saturation" in techniques:
# # #         processed_image = adjust_saturation(processed_image, float(params.get("saturation_factor", 1.0)))

# # #     # Noise Transformations
# # #     if "gaussian_noise" in techniques:
# # #         var = float(params.get("gaussian_variance", 0.01))
# # #         processed_image = add_gaussian_noise(processed_image, var=var)
# # #     if "salt_pepper_noise" in techniques:
# # #         amount = float(params.get("sap_amount", 0.005))
# # #         processed_image = add_salt_pepper_noise(processed_image, amount=amount)
# # #     if "speckle_noise" in techniques:
# # #         processed_image = add_speckle_noise(processed_image)
# # #     if "motion_blur" in techniques:
# # #         size = int(params.get("motion_blur_size", 9))
# # #         processed_image = add_motion_blur(processed_image, size=size)

# # #     # Occlusion Transformations
# # #     if "cutout" in techniques:
# # #         size = int(params.get("cutout_size", 50))
# # #         processed_image = apply_cutout(processed_image, size)
# # #     if "random_erasing" in techniques:
# # #         processed_image = apply_random_erasing(processed_image)

# # #     # Mixup and Cutmix
# # #     other_image_for_mix = None
# # #     if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset:
# # #         possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
# # #         if not possible_other_files and len(files_in_dataset) > 0:
# # #             possible_other_files = files_in_dataset
# # #         if possible_other_files:
# # #             other_filename = random.choice(possible_other_files)
# # #             other_filepath = os.path.join(source_dataset_folder, other_filename)
# # #             try:
# # #                 other_image_for_mix = Image.open(other_filepath)
# # #             except Exception as e:
# # #                 print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
# # #                 other_image_for_mix = None
# # #         else:
# # #             print("Warning: No other images available for mixup/cutmix.")

# # #     if "mixup" in techniques and other_image_for_mix:
# # #         alpha = float(params.get("mixup_alpha", 0.4))
# # #         processed_image = apply_mixup(processed_image, other_image_for_mix, alpha)
# # #     if "cutmix" in techniques and other_image_for_mix:
# # #         processed_image = apply_cutmix(processed_image, other_image_for_mix)
    
# # #     if other_image_for_mix: # Close the image if opened
# # #         other_image_for_mix.close()
# # #     return processed_image
# # # # --- End of Augmentation Functions ---


# # # @app.route('/upload', methods=['POST'])
# # # def upload_files():
# # #     dataset = request.form.get('dataset')
# # #     files = request.files.getlist('files')
# # #     if not dataset:
# # #         return jsonify({"error": "No dataset name provided"}), 400
# # #     if not files or all(f.filename == '' for f in files):
# # #         return jsonify({"error": "No files selected for upload"}), 400

# # #     dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
# # #     os.makedirs(dataset_folder, exist_ok=True)

# # #     for file in files:
# # #         if file.filename == '': continue # Should not happen if check above is done
# # #         filename = secure_filename(file.filename)
# # #         filepath = os.path.join(dataset_folder, filename)
# # #         file.save(filepath)
# # #         if filename.lower().endswith('.zip'):
# # #             try:
# # #                 with zipfile.ZipFile(filepath, 'r') as zip_ref:
# # #                     zip_ref.extractall(dataset_folder)
# # #                 os.remove(filepath)
# # #             except zipfile.BadZipFile:
# # #                 print(f"Bad zip file: {filepath}. Kept for inspection.")
# # #             except Exception as e:
# # #                 print(f"Error processing zip file {filepath}: {e}")

# # #     return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


# # # @app.route('/datasets', methods=['GET'])
# # # def list_datasets():
# # #     datasets_info = []
# # #     base_upload_folder = app.config['UPLOAD_FOLDER']
# # #     for folder_name in os.listdir(base_upload_folder):
# # #         folder_path = os.path.join(base_upload_folder, folder_name)
# # #         if os.path.isdir(folder_path):
# # #             try:
# # #                 files_in_folder = os.listdir(folder_path)
# # #                 image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# # #                 dataset_metadata = augmentation_metadata.get(folder_name, {})
# # #                 runs = dataset_metadata.get("augmentation_runs", [])
# # #                 datasets_info.append({
# # #                     'name': folder_name,
# # #                     'count': len(image_files),
# # #                     'files': image_files, # Only image files relevant for count and preview selection
# # #                     'all_files_in_folder': files_in_folder, # For listing if needed
# # #                     'augmentation_runs': runs
# # #                 })
# # #             except Exception as e:
# # #                 print(f"Error listing dataset {folder_name}: {e}")
# # #     return jsonify(datasets_info)

# # # @app.route('/augment', methods=['POST'])
# # # def augment_dataset_route():
# # #     data = request.get_json()
# # #     dataset_name = secure_filename(data.get('datasetName'))
# # #     techniques = data.get('techniques', [])
# # #     parameters = data.get('parameters', {})

# # #     if not dataset_name: return jsonify({'error': 'Dataset name not provided'}), 400
# # #     if not techniques: return jsonify({'error': 'No augmentation techniques selected'}), 400

# # #     source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
# # #     if not os.path.isdir(source_folder):
# # #         return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

# # #     dataset_augmented_base_path = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
# # #     os.makedirs(dataset_augmented_base_path, exist_ok=True)
# # #     run_index = 0
# # #     current_run_id = ""
# # #     target_run_folder_path = ""
# # #     while True:
# # #         current_run_id = f"run_{run_index}"
# # #         target_run_folder_path = os.path.join(dataset_augmented_base_path, current_run_id)
# # #         if not os.path.exists(target_run_folder_path):
# # #             os.makedirs(target_run_folder_path)
# # #             break
# # #         run_index += 1
# # #     zip_filename = f"{dataset_name}_augmented_{current_run_id}.zip"
# # #     zip_filepath = os.path.join(dataset_augmented_base_path, zip_filename)

# # #     source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# # #     augmented_file_names_for_zip = []

# # #     for filename in source_image_files:
# # #         img_path = os.path.join(source_folder, filename)
# # #         try:
# # #             with Image.open(img_path) as img:
# # #                 augmented_img = run_augmentations(
# # #                     img, techniques, parameters,
# # #                     source_folder, source_image_files, filename
# # #                 )
# # #                 if augmented_img.mode == 'P' and 'transparency' in augmented_img.info: augmented_img = augmented_img.convert("RGBA")
# # #                 elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): augmented_img = augmented_img.convert("RGBA")
# # #                 elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: augmented_img = augmented_img.convert('RGB')
# # #                 base, ext = os.path.splitext(filename)
# # #                 save_filename = f"aug_{base}{ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'}"
# # #                 save_path = os.path.join(target_run_folder_path, save_filename)
# # #                 if ext.lower() in ['.jpg', '.jpeg']: augmented_img.save(save_path, "JPEG", quality=95)
# # #                 else: augmented_img.save(save_path, "PNG")
# # #                 augmented_file_names_for_zip.append(save_filename)
# # #         except Exception as e: print(f"Error augmenting image {filename}: {e}")

# # #     with zipfile.ZipFile(zip_filepath, 'w') as zipf:
# # #         for aug_file_name in augmented_file_names_for_zip:
# # #             file_in_run_folder_path = os.path.join(target_run_folder_path, aug_file_name)
# # #             if os.path.isfile(file_in_run_folder_path):
# # #                  zipf.write(file_in_run_folder_path, aug_file_name)

# # #     if dataset_name not in augmentation_metadata: augmentation_metadata[dataset_name] = {"augmentation_runs": []}
# # #     elif "augmentation_runs" not in augmentation_metadata[dataset_name]: augmentation_metadata[dataset_name]["augmentation_runs"] = []
# # #     new_run_info = {
# # #         "run_id": current_run_id,
# # #         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
# # #         "techniques": techniques, "parameters": parameters,
# # #         "augmented_zip": zip_filename, "output_folder_name": current_run_id
# # #     }
# # #     augmentation_metadata[dataset_name]["augmentation_runs"].append(new_run_info)
# # #     with open(METADATA_FILE, 'w') as f: json.dump(augmentation_metadata, f, indent=4)
# # #     return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'run_id': current_run_id})

# # # # NEW PREVIEW ENDPOINT
# # # @app.route('/preview_augmentation', methods=['POST'])
# # # def preview_augmentation():
# # #     data = request.get_json()
# # #     dataset_name = secure_filename(data.get('datasetName'))
# # #     image_filename = secure_filename(data.get('imageFilename'))
# # #     technique = data.get('technique')
# # #     params = data.get('parameters', {}) # These are all params from frontend, pick specific ones

# # #     if not all([dataset_name, image_filename, technique]):
# # #         return jsonify({"error": "Missing data for preview"}), 400

# # #     original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
# # #     if not os.path.exists(original_image_path):
# # #         return jsonify({"error": "Sample image not found"}), 404

# # #     try:
# # #         with Image.open(original_image_path) as img:
# # #             preview_image = img.copy() # Start with a fresh copy for each preview

# # #             # Apply ONLY the single specified technique
# # #             if technique == "rotate": preview_image = rotate_image(preview_image, float(params.get("rotation_angle", 90)))
# # #             elif technique == "scale": preview_image = scale_image(preview_image, float(params.get("scaling_factor", 1.5)))
# # #             elif technique == "flip_horizontal": preview_image = flip_horizontal(preview_image)
# # #             elif technique == "flip_vertical": preview_image = flip_vertical(preview_image)
# # #             elif technique == "brightness": preview_image = adjust_brightness(preview_image, float(params.get("brightness_factor", 1.0)))
# # #             elif technique == "contrast": preview_image = adjust_contrast(preview_image, float(params.get("contrast_factor", 1.0)))
# # #             elif technique == "saturation": preview_image = adjust_saturation(preview_image, float(params.get("saturation_factor", 1.0)))
# # #             elif technique == "grayscale": preview_image = convert_grayscale(preview_image)
# # #             elif technique == "gaussian_noise": preview_image = add_gaussian_noise(preview_image, var=float(params.get("gaussian_variance", 0.01)))
# # #             elif technique == "salt_pepper_noise": preview_image = add_salt_pepper_noise(preview_image, amount=float(params.get("sap_amount", 0.005)))
# # #             elif technique == "speckle_noise": preview_image = add_speckle_noise(preview_image)
# # #             elif technique == "motion_blur": preview_image = add_motion_blur(preview_image, size=int(params.get("motion_blur_size", 9)))
# # #             elif technique == "cutout": preview_image = apply_cutout(preview_image, int(params.get("cutout_size", 50)))
# # #             elif technique == "random_erasing": preview_image = apply_random_erasing(preview_image)
# # #             # Mixup/Cutmix are not suitable for single image preview without a second image context here.
# # #             else:
# # #                 # If technique is unknown for preview, return original or error
# # #                 pass # Preview remains original if technique not handled for single preview

# # #             # Convert to base64
# # #             if preview_image.mode == 'P' and 'transparency' in preview_image.info: preview_image = preview_image.convert("RGBA")
# # #             elif preview_image.mode == 'LA' or (preview_image.mode == 'L' and 'transparency' in preview_image.info): preview_image = preview_image.convert("RGBA")
# # #             elif preview_image.mode not in ['RGB', 'RGBA', 'L']: preview_image = preview_image.convert('RGB')
            
# # #             buffered = io.BytesIO()
# # #             preview_image.save(buffered, format="PNG") # PNG is good for previews, supports transparency
# # #             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
# # #             return jsonify({"preview_image_base64": f"data:image/png;base64,{img_str}"})

# # #     except Exception as e:
# # #         print(f"Error generating preview for {technique}: {e}")
# # #         return jsonify({"error": f"Could not generate preview: {str(e)}"}), 500



# # # @app.route('/uploads/<dataset>/<filename>')
# # # def uploaded_file(dataset, filename):
# # #     return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

# # # @app.route('/augmented/<dataset>/<zipfilename>') 
# # # def serve_augmented_zip(dataset, zipfilename):
# # #     dataset_aug_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset))
# # #     return send_from_directory(dataset_aug_path, secure_filename(zipfilename), as_attachment=True)

# # # @app.route('/augmented_image/<dataset>/<run_id>/<filename>')
# # # def serve_augmented_image(dataset, run_id, filename):
# # #     image_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_id))
# # #     return send_from_directory(image_path, secure_filename(filename))

# # # if __name__ == '__main__':
# # #     app.run(host='0.0.0.0', port=5001, debug=True)



# # import os
# # import zipfile
# # import json
# # import random
# # import numpy as np
# # import cv2 # For motion blur
# # import datetime # For timestamps
# # import io # For image byte streaming
# # import base64 # For base64 encoding

# # from flask import Flask, request, jsonify, send_from_directory,send_file

# # from flask_cors import CORS
# # from werkzeug.utils import secure_filename
# # from PIL import Image, ImageOps, ImageEnhance, ImageChops

# # import os
# # from flask import Flask, request, jsonify
# # from gradio_client import Client, handle_file
# # from dotenv import load_dotenv
# # import shutil

# # # Load environment variables
# # load_dotenv()

# # GRADIO_URL = os.getenv("GRADIO_URL")

# # app = Flask(__name__)
# # CORS(app)

# # UPLOAD_FOLDER = 'uploads'
# # AUGMENTED_FOLDER = 'augmented'
# # METADATA_FILE = 'augmentation_metadata.json'
# # RESULT_DIR = "csv_results"
# # csv_DIR = "csv_uploads"

# # # Ensure results folder exists
# # os.makedirs(RESULT_DIR, exist_ok=True)
# # os.makedirs(csv_DIR, exist_ok=True)

# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # if os.path.exists(METADATA_FILE):
# #     with open(METADATA_FILE, 'r') as f:
# #         augmentation_metadata = json.load(f)
# # else:
# #     augmentation_metadata = {}

# # # --- Start of Augmentation Functions (Identical to previous full version) ---
# # # Geometric Transformations
# # def rotate_image(image, angle=90):
# #     return image.rotate(angle, expand=True)

# # def scale_image(image, scale=1.5):
# #     if scale <= 0: return image 
# #     width, height = image.size
# #     return image.resize((int(width * scale), int(height * scale)))

# # def translate_image(image, x_offset, y_offset):
# #     return ImageChops.offset(image, x_offset, y_offset)

# # def flip_horizontal(image):
# #     return ImageOps.mirror(image)

# # def flip_vertical(image):
# #     return ImageOps.flip(image)

# # def crop_image(image, left, top, right, bottom):
# #     width, height = image.size
# #     left = max(0, int(left))
# #     top = max(0, int(top))
# #     right = min(width, int(right))
# #     bottom = min(height, int(bottom))
# #     if left < right and top < bottom:
# #         return image.crop((left, top, right, bottom))
# #     return image

# # def pad_image(image, padding_size, padding_color="#000000"):
# #     padding_size = int(padding_size)
# #     if padding_size <=0: return image
# #     return ImageOps.expand(image, border=padding_size, fill=padding_color)

# # # Color Transformations
# # def adjust_brightness(image, factor):
# #     enhancer = ImageEnhance.Brightness(image.convert('RGB'))
# #     return enhancer.enhance(factor)

# # def adjust_contrast(image, factor):
# #     enhancer = ImageEnhance.Contrast(image.convert('RGB'))
# #     return enhancer.enhance(factor)

# # def convert_grayscale(image):
# #     return ImageOps.grayscale(image).convert('RGB')

# # def adjust_saturation(image, factor):
# #     enhancer = ImageEnhance.Color(image.convert('RGB'))
# #     return enhancer.enhance(factor)

# # # Noise Transformations
# # def add_gaussian_noise(image, mean=0, var=0.01):
# #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# #     noise = np.random.normal(mean, var ** 0.5, img_array.shape)
# #     img_noisy = img_array + noise
# #     img_noisy = np.clip(img_noisy, 0, 1)
# #     img_noisy = (img_noisy * 255).astype(np.uint8)
# #     return Image.fromarray(img_noisy)

# # def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
# #     img_array = np.array(image.convert('RGB'))
# #     num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
# #     num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)
# #     if img_array.size > 0 : # Check if image is not empty
# #         coords_salt = [np.random.randint(0, i - 1 if i > 1 else 1, num_salt) for i in img_array.shape[:2]]
# #         if num_salt > 0 : img_array[coords_salt[0], coords_salt[1], :] = [255,255,255]
# #         coords_pepper = [np.random.randint(0, i - 1 if i > 1 else 1, num_pepper) for i in img_array.shape[:2]]
# #         if num_pepper > 0: img_array[coords_pepper[0], coords_pepper[1], :] = [0,0,0]
# #     return Image.fromarray(img_array)

# # def add_speckle_noise(image):
# #     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
# #     noise = np.random.randn(*img_array.shape)
# #     img_noisy = img_array + img_array * noise
# #     img_noisy = np.clip(img_noisy, 0, 1)
# #     img_noisy = (img_noisy * 255).astype(np.uint8)
# #     return Image.fromarray(img_noisy)

# # def add_motion_blur(image, size=9):
# #     size = int(size)
# #     if size <= 1 or size % 2 == 0 : size = 3 # Ensure odd kernel size > 1
# #     kernel = np.zeros((size, size))
# #     kernel[int((size - 1)/2), :] = np.ones(size)
# #     kernel = kernel / size
# #     img_array = np.array(image.convert('RGB'))
# #     img_blur = cv2.filter2D(img_array, -1, kernel)
# #     return Image.fromarray(img_blur)

# # # Occlusion Transformations
# # def apply_cutout(image, mask_size):
# #     mask_size = int(mask_size)
# #     if mask_size <=0: return image
# #     img_array = np.array(image.convert('RGB'))
# #     h, w = img_array.shape[:2]
# #     if h == 0 or w == 0: return image
# #     y = np.random.randint(h)
# #     x = np.random.randint(w)
# #     y1 = np.clip(y - mask_size // 2, 0, h)
# #     y2 = np.clip(y + mask_size // 2, 0, h)
# #     x1 = np.clip(x - mask_size // 2, 0, w)
# #     x2 = np.clip(x + mask_size // 2, 0, w)
# #     img_array[y1:y2, x1:x2] = 0
# #     return Image.fromarray(img_array)

# # def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
# #     img_array = np.array(image.convert('RGB'))
# #     h, w, c = img_array.shape
# #     s_img = h * w
# #     s_erase = np.random.uniform(sl, sh) * s_img
# #     r_aspect = np.random.uniform(r1, 1/r1)
# #     h_e = int(np.sqrt(s_erase * r_aspect))
# #     w_e = int(np.sqrt(s_erase / r_aspect))
# #     if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h:
# #         return Image.fromarray(img_array)
# #     x_e = np.random.randint(0, w - w_e + 1)
# #     y_e = np.random.randint(0, h - h_e + 1)
# #     img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
# #     return Image.fromarray(img_array)

# # # Mix Transformations
# # def apply_mixup(image, other_image, alpha=0.4):
# #     if other_image is None: return image
# #     lam = np.random.beta(alpha, alpha)
# #     image_array = np.array(image.convert('RGB')).astype(np.float32)
# #     other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
# #     mixed_array = lam * image_array + (1 - lam) * other_array
# #     return Image.fromarray(mixed_array.astype(np.uint8))

# # def apply_cutmix(image, other_image):
# #     if other_image is None: return image
# #     img_array = np.array(image.convert('RGB'))
# #     other_array = np.array(other_image.convert('RGB').resize(image.size))
# #     h, w, _ = img_array.shape
# #     if h == 0 or w == 0: return image
# #     lam = np.random.beta(1.0, 1.0)
# #     cut_ratio = np.sqrt(1. - lam)
# #     cut_w = int(w * cut_ratio)
# #     cut_h = int(h * cut_ratio)
# #     if cut_w == 0 or cut_h == 0: return image
# #     cx = np.random.randint(w)
# #     cy = np.random.randint(h)
# #     bbx1 = np.clip(cx - cut_w // 2, 0, w)
# #     bby1 = np.clip(cy - cut_h // 2, 0, h)
# #     bbx2 = np.clip(cx + cut_w // 2, 0, w)
# #     bby2 = np.clip(cy + cut_h // 2, 0, h)
# #     if bbx1 < bbx2 and bby1 < bby2:
# #       img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
# #     return Image.fromarray(img_array)

# # # Master Augmentation Application Function
# # def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
# #     processed_image = image.copy() # Work on a copy
# #     # Geometric Transformations
# #     if "rotate" in techniques:
# #         angle = float(params.get("rotation_angle", 90))
# #         processed_image = rotate_image(processed_image, angle)
# #     if "scale" in techniques:
# #         scale_factor = float(params.get("scaling_factor", 1.5))
# #         processed_image = scale_image(processed_image, scale_factor)
# #     # ... (all other transformations from previous version) ...
# #     if "translate" in techniques:
# #         x_offset = int(params.get("translation_x", 0))
# #         y_offset = int(params.get("translation_y", 0))
# #         processed_image = translate_image(processed_image, x_offset, y_offset)
# #     if "flip_horizontal" in techniques:
# #         processed_image = flip_horizontal(processed_image)
# #     if "flip_vertical" in techniques:
# #         processed_image = flip_vertical(processed_image)
# #     if "crop" in techniques:
# #         left = int(params.get("crop_left", 0))
# #         top = int(params.get("crop_top", 0))
# #         right = int(params.get("crop_right", processed_image.width if processed_image else 0))
# #         bottom = int(params.get("crop_bottom", processed_image.height if processed_image else 0))
# #         processed_image = crop_image(processed_image, left, top, right, bottom)
# #     if "pad" in techniques:
# #         padding = int(params.get("padding_size", 0))
# #         padding_color = params.get("padding_color", "#000000")
# #         processed_image = pad_image(processed_image, padding, padding_color)

# #     # Color Transformations
# #     if "brightness" in techniques:
# #         processed_image = adjust_brightness(processed_image, float(params.get("brightness_factor", 1.0)))
# #     if "contrast" in techniques:
# #         processed_image = adjust_contrast(processed_image, float(params.get("contrast_factor", 1.0)))
# #     if "grayscale" in techniques:
# #         processed_image = convert_grayscale(processed_image)
# #     if "saturation" in techniques:
# #         processed_image = adjust_saturation(processed_image, float(params.get("saturation_factor", 1.0)))

# #     # Noise Transformations
# #     if "gaussian_noise" in techniques:
# #         var = float(params.get("gaussian_variance", 0.01))
# #         processed_image = add_gaussian_noise(processed_image, var=var)
# #     if "salt_pepper_noise" in techniques:
# #         amount = float(params.get("sap_amount", 0.005))
# #         processed_image = add_salt_pepper_noise(processed_image, amount=amount)
# #     if "speckle_noise" in techniques:
# #         processed_image = add_speckle_noise(processed_image)
# #     if "motion_blur" in techniques:
# #         size = int(params.get("motion_blur_size", 9))
# #         processed_image = add_motion_blur(processed_image, size=size)

# #     # Occlusion Transformations
# #     if "cutout" in techniques:
# #         size = int(params.get("cutout_size", 50))
# #         processed_image = apply_cutout(processed_image, size)
# #     if "random_erasing" in techniques:
# #         processed_image = apply_random_erasing(processed_image)

# #     # Mixup and Cutmix
# #     other_image_for_mix = None
# #     if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset:
# #         possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
# #         if not possible_other_files and len(files_in_dataset) > 0:
# #             possible_other_files = files_in_dataset
# #         if possible_other_files:
# #             other_filename = random.choice(possible_other_files)
# #             other_filepath = os.path.join(source_dataset_folder, other_filename)
# #             try:
# #                 other_image_for_mix = Image.open(other_filepath)
# #             except Exception as e:
# #                 print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
# #                 other_image_for_mix = None
# #         else:
# #             print("Warning: No other images available for mixup/cutmix.")

# #     if "mixup" in techniques and other_image_for_mix:
# #         alpha = float(params.get("mixup_alpha", 0.4))
# #         processed_image = apply_mixup(processed_image, other_image_for_mix, alpha)
# #     if "cutmix" in techniques and other_image_for_mix:
# #         processed_image = apply_cutmix(processed_image, other_image_for_mix)
    
# #     if other_image_for_mix: # Close the image if opened
# #         other_image_for_mix.close()
# #     return processed_image
# # # --- End of Augmentation Functions ---


# # @app.route('/upload', methods=['POST'])
# # def upload_files():
# #     dataset = request.form.get('dataset')
# #     files = request.files.getlist('files')
# #     if not dataset:
# #         return jsonify({"error": "No dataset name provided"}), 400
# #     if not files or all(f.filename == '' for f in files):
# #         return jsonify({"error": "No files selected for upload"}), 400

# #     dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
# #     os.makedirs(dataset_folder, exist_ok=True)

# #     for file in files:
# #         if file.filename == '': continue # Should not happen if check above is done
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(dataset_folder, filename)
# #         file.save(filepath)
# #         if filename.lower().endswith('.zip'):
# #             try:
# #                 with zipfile.ZipFile(filepath, 'r') as zip_ref:
# #                     zip_ref.extractall(dataset_folder)
# #                 os.remove(filepath)
# #             except zipfile.BadZipFile:
# #                 print(f"Bad zip file: {filepath}. Kept for inspection.")
# #             except Exception as e:
# #                 print(f"Error processing zip file {filepath}: {e}")

# #     return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


# # @app.route('/datasets', methods=['GET'])
# # def list_datasets():
# #     datasets_info = []
# #     base_upload_folder = app.config['UPLOAD_FOLDER']
# #     for folder_name in os.listdir(base_upload_folder):
# #         folder_path = os.path.join(base_upload_folder, folder_name)
# #         if os.path.isdir(folder_path):
# #             try:
# #                 files_in_folder = os.listdir(folder_path)
# #                 image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# #                 dataset_metadata = augmentation_metadata.get(folder_name, {})
# #                 runs = dataset_metadata.get("augmentation_runs", [])
# #                 datasets_info.append({
# #                     'name': folder_name,
# #                     'count': len(image_files),
# #                     'files': image_files, # Only image files relevant for count and preview selection
# #                     'all_files_in_folder': files_in_folder, # For listing if needed
# #                     'augmentation_runs': runs
# #                 })
# #             except Exception as e:
# #                 print(f"Error listing dataset {folder_name}: {e}")
# #     return jsonify(datasets_info)

# # @app.route('/augment', methods=['POST'])
# # def augment_dataset_route():
# #     data = request.get_json()
# #     dataset_name = secure_filename(data.get('datasetName'))
# #     techniques = data.get('techniques', [])
# #     parameters = data.get('parameters', {})

# #     if not dataset_name: return jsonify({'error': 'Dataset name not provided'}), 400
# #     if not techniques: return jsonify({'error': 'No augmentation techniques selected'}), 400

# #     source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
# #     if not os.path.isdir(source_folder):
# #         return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

# #     dataset_augmented_base_path = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
# #     os.makedirs(dataset_augmented_base_path, exist_ok=True)
# #     run_index = 0
# #     current_run_id = ""
# #     target_run_folder_path = ""
# #     while True:
# #         current_run_id = f"run_{run_index}"
# #         target_run_folder_path = os.path.join(dataset_augmented_base_path, current_run_id)
# #         if not os.path.exists(target_run_folder_path):
# #             os.makedirs(target_run_folder_path)
# #             break
# #         run_index += 1
# #     zip_filename = f"{dataset_name}_augmented_{current_run_id}.zip"
# #     zip_filepath = os.path.join(dataset_augmented_base_path, zip_filename)

# #     source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# #     augmented_file_names_for_zip = []

# #     for filename in source_image_files:
# #         img_path = os.path.join(source_folder, filename)
# #         try:
# #             with Image.open(img_path) as img:
# #                 augmented_img = run_augmentations(
# #                     img, techniques, parameters,
# #                     source_folder, source_image_files, filename
# #                 )
# #                 if augmented_img.mode == 'P' and 'transparency' in augmented_img.info: augmented_img = augmented_img.convert("RGBA")
# #                 elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): augmented_img = augmented_img.convert("RGBA")
# #                 elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: augmented_img = augmented_img.convert('RGB')
# #                 base, ext = os.path.splitext(filename)
# #                 save_filename = f"aug_{base}{ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'}"
# #                 save_path = os.path.join(target_run_folder_path, save_filename)
# #                 if ext.lower() in ['.jpg', '.jpeg']: augmented_img.save(save_path, "JPEG", quality=95)
# #                 else: augmented_img.save(save_path, "PNG")
# #                 augmented_file_names_for_zip.append(save_filename)
# #         except Exception as e: print(f"Error augmenting image {filename}: {e}")

# #     with zipfile.ZipFile(zip_filepath, 'w') as zipf:
# #         for aug_file_name in augmented_file_names_for_zip:
# #             file_in_run_folder_path = os.path.join(target_run_folder_path, aug_file_name)
# #             if os.path.isfile(file_in_run_folder_path):
# #                  zipf.write(file_in_run_folder_path, aug_file_name)

# #     if dataset_name not in augmentation_metadata: augmentation_metadata[dataset_name] = {"augmentation_runs": []}
# #     elif "augmentation_runs" not in augmentation_metadata[dataset_name]: augmentation_metadata[dataset_name]["augmentation_runs"] = []
# #     new_run_info = {
# #         "run_id": current_run_id,
# #         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
# #         "techniques": techniques, "parameters": parameters,
# #         "augmented_zip": zip_filename, "output_folder_name": current_run_id
# #     }
# #     augmentation_metadata[dataset_name]["augmentation_runs"].append(new_run_info)
# #     with open(METADATA_FILE, 'w') as f: json.dump(augmentation_metadata, f, indent=4)
# #     return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'run_id': current_run_id})

# # # NEW PREVIEW ENDPOINT
# # @app.route('/preview_augmentation', methods=['POST'])
# # def preview_augmentation():
# #     data = request.get_json()
# #     dataset_name = secure_filename(data.get('datasetName'))
# #     image_filename = secure_filename(data.get('imageFilename'))
# #     technique = data.get('technique')
# #     params = data.get('parameters', {}) # These are all params from frontend, pick specific ones

# #     if not all([dataset_name, image_filename, technique]):
# #         return jsonify({"error": "Missing data for preview"}), 400

# #     original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
# #     if not os.path.exists(original_image_path):
# #         return jsonify({"error": "Sample image not found"}), 404

# #     try:
# #         with Image.open(original_image_path) as img:
# #             preview_image = img.copy() # Start with a fresh copy for each preview

# #             # Apply ONLY the single specified technique
# #             if technique == "rotate": preview_image = rotate_image(preview_image, float(params.get("rotation_angle", 90)))
# #             elif technique == "scale": preview_image = scale_image(preview_image, float(params.get("scaling_factor", 1.5)))
# #             elif technique == "flip_horizontal": preview_image = flip_horizontal(preview_image)
# #             elif technique == "flip_vertical": preview_image = flip_vertical(preview_image)
# #             elif technique == "brightness": preview_image = adjust_brightness(preview_image, float(params.get("brightness_factor", 1.0)))
# #             elif technique == "contrast": preview_image = adjust_contrast(preview_image, float(params.get("contrast_factor", 1.0)))
# #             elif technique == "saturation": preview_image = adjust_saturation(preview_image, float(params.get("saturation_factor", 1.0)))
# #             elif technique == "grayscale": preview_image = convert_grayscale(preview_image)
# #             elif technique == "gaussian_noise": preview_image = add_gaussian_noise(preview_image, var=float(params.get("gaussian_variance", 0.01)))
# #             elif technique == "salt_pepper_noise": preview_image = add_salt_pepper_noise(preview_image, amount=float(params.get("sap_amount", 0.005)))
# #             elif technique == "speckle_noise": preview_image = add_speckle_noise(preview_image)
# #             elif technique == "motion_blur": preview_image = add_motion_blur(preview_image, size=int(params.get("motion_blur_size", 9)))
# #             elif technique == "cutout": preview_image = apply_cutout(preview_image, int(params.get("cutout_size", 50)))
# #             elif technique == "random_erasing": preview_image = apply_random_erasing(preview_image)
# #             # Mixup/Cutmix are not suitable for single image preview without a second image context here.
# #             else:
# #                 # If technique is unknown for preview, return original or error
# #                 pass # Preview remains original if technique not handled for single preview

# #             # Convert to base64
# #             if preview_image.mode == 'P' and 'transparency' in preview_image.info: preview_image = preview_image.convert("RGBA")
# #             elif preview_image.mode == 'LA' or (preview_image.mode == 'L' and 'transparency' in preview_image.info): preview_image = preview_image.convert("RGBA")
# #             elif preview_image.mode not in ['RGB', 'RGBA', 'L']: preview_image = preview_image.convert('RGB')
            
# #             buffered = io.BytesIO()
# #             preview_image.save(buffered, format="PNG") # PNG is good for previews, supports transparency
# #             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
# #             return jsonify({"preview_image_base64": f"data:image/png;base64,{img_str}"})

# #     except Exception as e:
# #         print(f"Error generating preview for {technique}: {e}")
# #         return jsonify({"error": f"Could not generate preview: {str(e)}"}), 500


# # @app.route('/uploads/<dataset>/<filename>')
# # def uploaded_file(dataset, filename):
# #     return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

# # @app.route('/augmented/<dataset>/<zipfilename>') 
# # def serve_augmented_zip(dataset, zipfilename):
# #     dataset_aug_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset))
# #     return send_from_directory(dataset_aug_path, secure_filename(zipfilename), as_attachment=True)

# # @app.route('/augmented_image/<dataset>/<run_id>/<filename>')
# # def serve_augmented_image(dataset, run_id, filename):
# #     image_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_id))
# #     return send_from_directory(image_path, secure_filename(filename))


# # @app.route('/generate-synthetic', methods=['POST'])
# # def generate_synthetic():
# #     file = request.files.get('file')
# #     epochs = int(request.form.get('epochs', 5))
# #     num_samples = int(request.form.get('samples', 100))

# #     if not file:
# #         return jsonify({"error": "No file uploaded"}), 400

# #     # Save temp uploaded file
# #     temp_input_path = os.path.join(csv_DIR, file.filename)
# #     file.save(temp_input_path)

# #     try:
# #         # Call CTGAN via Gradio client
# #         client = Client(GRADIO_URL)
# #         result = client.predict(
# #             handle_file(temp_input_path),
# #             epochs,
# #             num_samples
# #         )

# #         # Save to results directory
# #         output_filename = file.filename.replace(".csv", "_gen.csv")
# #         output_path = os.path.join(RESULT_DIR, output_filename)
# #         shutil.copy(result, output_path)

# #         return jsonify({
# #             "output_file": output_filename
# #         })

# #     except Exception as e:
# #         return jsonify({"error": str(e)}),500


# # @app.route('/download-csv/<filename>', methods=['GET'])
# # def download_file(filename):
# #     filepath = os.path.join(RESULT_DIR, filename)
# #     return send_file(filepath, as_attachment=True)


# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5001, debug=True)



# import os
# import zipfile
# import json
# import random
# import numpy as np
# import cv2 # For motion blur
# import datetime # For timestamps
# import io # For image byte streaming
# import base64 # For base64 encoding

# from flask import Flask, request, jsonify, send_from_directory,send_file

# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from PIL import Image, ImageOps, ImageEnhance, ImageChops

# # import os # Already imported
# # from flask import Flask, request, jsonify # Already imported
# from gradio_client import Client, handle_file
# from dotenv import load_dotenv
# import shutil

# # Load environment variables
# load_dotenv()

# GRADIO_URL = os.getenv("GRADIO_URL")

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = 'uploads'
# AUGMENTED_FOLDER = 'augmented'
# METADATA_FILE = 'augmentation_metadata.json'
# RESULT_DIR = "csv_results"
# csv_DIR = "csv_uploads"

# # Ensure results folder exists
# os.makedirs(RESULT_DIR, exist_ok=True)
# os.makedirs(csv_DIR, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# if os.path.exists(METADATA_FILE):
#     with open(METADATA_FILE, 'r') as f:
#         augmentation_metadata = json.load(f)
# else:
#     augmentation_metadata = {}

# # --- Start of Augmentation Functions (Identical to previous full version) ---
# # Geometric Transformations
# def rotate_image(image, angle=90):
#     return image.rotate(angle, expand=True)

# def scale_image(image, scale=1.5):
#     if scale <= 0: return image
#     width, height = image.size
#     return image.resize((int(width * scale), int(height * scale)))

# def translate_image(image, x_offset, y_offset):
#     return ImageChops.offset(image, x_offset, y_offset)

# def flip_horizontal(image):
#     return ImageOps.mirror(image)

# def flip_vertical(image):
#     return ImageOps.flip(image)

# def crop_image(image, left, top, right, bottom):
#     width, height = image.size
#     # Assuming parameters are percentages of width/height if they are float and < 1, or direct pixels
#     # For now, let's assume they are direct pixel values as per original implementation structure
#     # The frontend sends numbers, PIL expects pixels.
#     # The backend should ensure these are integers after any calculation if they were %
    
#     # Ensure crop coordinates are integers and within image bounds
#     left = max(0, int(left))
#     top = max(0, int(top))
#     right = min(width, int(right))
#     bottom = min(height, int(bottom))

#     if left < right and top < bottom:
#         return image.crop((left, top, right, bottom))
#     return image # Return original if crop dimensions are invalid

# def pad_image(image, padding_size, padding_color="#000000"):
#     padding_size = int(padding_size)
#     if padding_size <=0: return image
#     return ImageOps.expand(image, border=padding_size, fill=padding_color)

# # Color Transformations
# def adjust_brightness(image, factor):
#     enhancer = ImageEnhance.Brightness(image.convert('RGB'))
#     return enhancer.enhance(factor)

# def adjust_contrast(image, factor):
#     enhancer = ImageEnhance.Contrast(image.convert('RGB'))
#     return enhancer.enhance(factor)

# def convert_grayscale(image):
#     return ImageOps.grayscale(image).convert('RGB')

# def adjust_saturation(image, factor):
#     enhancer = ImageEnhance.Color(image.convert('RGB'))
#     return enhancer.enhance(factor)

# # Noise Transformations
# def add_gaussian_noise(image, mean=0, var=0.01):
#     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
#     noise = np.random.normal(mean, var ** 0.5, img_array.shape)
#     img_noisy = img_array + noise
#     img_noisy = np.clip(img_noisy, 0, 1)
#     img_noisy = (img_noisy * 255).astype(np.uint8)
#     return Image.fromarray(img_noisy)

# def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
#     img_array = np.array(image.convert('RGB'))
#     num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
#     num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)
#     if img_array.size > 0 : # Check if image is not empty
#         # Ensure indices are within bounds, especially for small images
#         if num_salt > 0:
#             salt_rows = np.random.randint(0, max(1, img_array.shape[0]), num_salt)
#             salt_cols = np.random.randint(0, max(1, img_array.shape[1]), num_salt)
#             img_array[salt_rows, salt_cols, :] = [255,255,255]
#         if num_pepper > 0:
#             pepper_rows = np.random.randint(0, max(1, img_array.shape[0]), num_pepper)
#             pepper_cols = np.random.randint(0, max(1, img_array.shape[1]), num_pepper)
#             img_array[pepper_rows, pepper_cols, :] = [0,0,0]
#     return Image.fromarray(img_array)

# def add_speckle_noise(image):
#     img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
#     noise = np.random.randn(*img_array.shape)
#     img_noisy = img_array + img_array * noise
#     img_noisy = np.clip(img_noisy, 0, 1)
#     img_noisy = (img_noisy * 255).astype(np.uint8)
#     return Image.fromarray(img_noisy)

# def add_motion_blur(image, size=9):
#     size = int(size)
#     if size <= 1: size = 3 # Ensure kernel size > 1
#     if size % 2 == 0 : size +=1 # Ensure odd kernel size
#     kernel = np.zeros((size, size))
#     kernel[int((size - 1)/2), :] = np.ones(size)
#     kernel = kernel / size
#     img_array = np.array(image.convert('RGB'))
#     img_blur = cv2.filter2D(img_array, -1, kernel)
#     return Image.fromarray(img_blur)

# # Occlusion Transformations
# def apply_cutout(image, mask_size):
#     mask_size = int(mask_size)
#     if mask_size <=0: return image
#     img_array = np.array(image.convert('RGB'))
#     h, w = img_array.shape[:2]
#     if h == 0 or w == 0: return image
#     y = np.random.randint(h)
#     x = np.random.randint(w)
#     y1 = np.clip(y - mask_size // 2, 0, h)
#     y2 = np.clip(y + mask_size // 2, 0, h)
#     x1 = np.clip(x - mask_size // 2, 0, w)
#     x2 = np.clip(x + mask_size // 2, 0, w)
#     img_array[y1:y2, x1:x2] = 0
#     return Image.fromarray(img_array)

# def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
#     img_array = np.array(image.convert('RGB'))
#     h, w, c = img_array.shape
#     s_img = h * w
#     if s_img == 0: return image # Avoid division by zero for empty images
#     s_erase = np.random.uniform(sl, sh) * s_img
#     r_aspect = np.random.uniform(r1, 1/r1 if r1 != 0 else 1) # Avoid division by zero for r1
    
#     h_e = int(np.sqrt(s_erase * r_aspect))
#     w_e = int(np.sqrt(s_erase / r_aspect if r_aspect != 0 else s_erase)) # Avoid division by zero

#     if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h: # Check if erase dimensions are valid
#         return Image.fromarray(img_array)
        
#     x_e = np.random.randint(0, w - w_e + 1)
#     y_e = np.random.randint(0, h - h_e + 1)
#     img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
#     return Image.fromarray(img_array)

# # Mix Transformations
# def apply_mixup(image, other_image, alpha=0.4):
#     if other_image is None: return image
#     lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0 # Ensure alpha > 0 for beta distribution
#     image_array = np.array(image.convert('RGB')).astype(np.float32)
#     # Resize other_image to match image's size before converting to array
#     other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
#     mixed_array = lam * image_array + (1 - lam) * other_array
#     return Image.fromarray(mixed_array.astype(np.uint8))

# def apply_cutmix(image, other_image):
#     if other_image is None: return image
#     img_array = np.array(image.convert('RGB'))
#     # Resize other_image to match image's size before converting to array
#     other_array = np.array(other_image.convert('RGB').resize(image.size))
    
#     h, w, _ = img_array.shape
#     if h == 0 or w == 0: return image
    
#     # Ensure beta(1.0, 1.0) which is uniform(0,1)
#     lam = np.random.beta(1.0, 1.0)
#     cut_ratio = np.sqrt(1. - lam)
#     cut_w = int(w * cut_ratio)
#     cut_h = int(h * cut_ratio)

#     if cut_w == 0 or cut_h == 0: return image # No cut to make

#     cx = np.random.randint(w)
#     cy = np.random.randint(h)

#     bbx1 = np.clip(cx - cut_w // 2, 0, w)
#     bby1 = np.clip(cy - cut_h // 2, 0, h)
#     bbx2 = np.clip(cx + cut_w // 2, 0, w)
#     bby2 = np.clip(cy + cut_h // 2, 0, h)

#     if bbx1 < bbx2 and bby1 < bby2: # Ensure the patch has a valid area
#       img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
#     return Image.fromarray(img_array)

# # Master Augmentation Application Function
# def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
#     processed_image = image.copy() # Work on a copy
    
#     # Geometric Transformations
#     if "rotate" in techniques:
#         angle = float(params.get("rotation_angle", 90))
#         processed_image = rotate_image(processed_image, angle)
#     if "scale" in techniques:
#         scale_factor = float(params.get("scaling_factor", 1.5))
#         processed_image = scale_image(processed_image, scale_factor)
#     if "translate" in techniques:
#         x_offset = int(params.get("translation_x", 0))
#         y_offset = int(params.get("translation_y", 0))
#         processed_image = translate_image(processed_image, x_offset, y_offset)
#     if "flip_horizontal" in techniques:
#         processed_image = flip_horizontal(processed_image)
#     if "flip_vertical" in techniques:
#         processed_image = flip_vertical(processed_image)
#     if "crop" in techniques:
#         # For crop, parameters are absolute pixel values
#         img_w, img_h = processed_image.size
#         left = float(params.get("crop_left", 0))
#         top = float(params.get("crop_top", 0))
#         right = float(params.get("crop_right", img_w))
#         bottom = float(params.get("crop_bottom", img_h))
#         processed_image = crop_image(processed_image, left, top, right, bottom)
#     if "pad" in techniques:
#         padding = int(params.get("padding_size", 0))
#         padding_color = params.get("padding_color", "#000000")
#         processed_image = pad_image(processed_image, padding, padding_color)

#     # Color Transformations
#     if "brightness" in techniques:
#         processed_image = adjust_brightness(processed_image, float(params.get("brightness_factor", 1.0)))
#     if "contrast" in techniques:
#         processed_image = adjust_contrast(processed_image, float(params.get("contrast_factor", 1.0)))
#     if "grayscale" in techniques:
#         processed_image = convert_grayscale(processed_image)
#     if "saturation" in techniques:
#         processed_image = adjust_saturation(processed_image, float(params.get("saturation_factor", 1.0)))

#     # Noise Transformations
#     if "gaussian_noise" in techniques:
#         var = float(params.get("gaussian_variance", 0.01))
#         processed_image = add_gaussian_noise(processed_image, var=var)
#     if "salt_pepper_noise" in techniques:
#         amount = float(params.get("sap_amount", 0.005))
#         processed_image = add_salt_pepper_noise(processed_image, amount=amount)
#     if "speckle_noise" in techniques:
#         processed_image = add_speckle_noise(processed_image)
#     if "motion_blur" in techniques:
#         size = int(params.get("motion_blur_size", 9))
#         processed_image = add_motion_blur(processed_image, size=size)

#     # Occlusion Transformations
#     if "cutout" in techniques:
#         size = int(params.get("cutout_size", 50))
#         processed_image = apply_cutout(processed_image, size)
#     if "random_erasing" in techniques:
#         # Default params for random_erasing are in its definition
#         processed_image = apply_random_erasing(processed_image)

#     # Mixup and Cutmix (require other images from the dataset)
#     other_image_for_mix = None
#     if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset and source_dataset_folder:
#         possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
#         if not possible_other_files and len(files_in_dataset) > 0: # If only one image, mix with itself
#             possible_other_files = files_in_dataset 
        
#         if possible_other_files:
#             other_filename = random.choice(possible_other_files)
#             other_filepath = os.path.join(source_dataset_folder, other_filename)
#             try:
#                 other_image_for_mix = Image.open(other_filepath)
#             except Exception as e:
#                 print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
#                 other_image_for_mix = None # Fallback: do not apply if other image fails to load
#         # else: # No other images or current image itself to mix with (should not happen if files_in_dataset is not empty)
#             # print("Warning: No other images available for mixup/cutmix, even current image not found.")

#     if "mixup" in techniques: # No need to check other_image_for_mix here, apply_mixup handles it
#         alpha = float(params.get("mixup_alpha", 0.4))
#         processed_image = apply_mixup(processed_image, other_image_for_mix, alpha)
#     if "cutmix" in techniques: # No need to check other_image_for_mix here, apply_cutmix handles it
#         processed_image = apply_cutmix(processed_image, other_image_for_mix)
    
#     if other_image_for_mix: # Close the image if opened
#         other_image_for_mix.close()
        
#     return processed_image
# # --- End of Augmentation Functions ---


# @app.route('/upload', methods=['POST'])
# def upload_files():
#     dataset = request.form.get('dataset')
#     files = request.files.getlist('files')
#     if not dataset:
#         return jsonify({"error": "No dataset name provided"}), 400
#     if not files or all(f.filename == '' for f in files):
#         return jsonify({"error": "No files selected for upload"}), 400

#     dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
#     os.makedirs(dataset_folder, exist_ok=True)

#     for file in files:
#         if file.filename == '': continue 
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(dataset_folder, filename)
#         file.save(filepath)
#         if filename.lower().endswith('.zip'):
#             try:
#                 with zipfile.ZipFile(filepath, 'r') as zip_ref:
#                     zip_ref.extractall(dataset_folder)
#                 os.remove(filepath) 
#             except zipfile.BadZipFile:
#                 print(f"Bad zip file: {filepath}. Kept for inspection.")
#             except Exception as e:
#                 print(f"Error processing zip file {filepath}: {e}")
#                 # Optionally keep the bad zip for inspection or remove as well
#                 # os.remove(filepath) 

#     return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


# @app.route('/datasets', methods=['GET'])
# def list_datasets():
#     datasets_info = []
#     base_upload_folder = app.config['UPLOAD_FOLDER']
#     for folder_name in os.listdir(base_upload_folder):
#         folder_path = os.path.join(base_upload_folder, folder_name)
#         if os.path.isdir(folder_path):
#             try:
#                 files_in_folder = os.listdir(folder_path)
#                 image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#                 dataset_metadata = augmentation_metadata.get(folder_name, {})
#                 runs = dataset_metadata.get("augmentation_runs", [])
#                 datasets_info.append({
#                     'name': folder_name,
#                     'count': len(image_files),
#                     'files': image_files, 
#                     'all_files_in_folder': files_in_folder, 
#                     'augmentation_runs': runs
#                 })
#             except Exception as e:
#                 print(f"Error listing dataset {folder_name}: {e}")
#     return jsonify(datasets_info)

# @app.route('/augment', methods=['POST'])
# def augment_dataset_route():
#     data = request.get_json()
#     dataset_name = secure_filename(data.get('datasetName'))
#     techniques = data.get('techniques', [])
#     parameters = data.get('parameters', {})

#     if not dataset_name: return jsonify({'error': 'Dataset name not provided'}), 400
#     if not techniques: return jsonify({'error': 'No augmentation techniques selected'}), 400

#     source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
#     if not os.path.isdir(source_folder):
#         return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

#     dataset_augmented_base_path = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
#     os.makedirs(dataset_augmented_base_path, exist_ok=True)
#     run_index = 0
#     current_run_id = ""
#     target_run_folder_path = ""
#     # Find the next available run_id
#     existing_runs = augmentation_metadata.get(dataset_name, {}).get("augmentation_runs", [])
#     existing_run_ids = {int(r["run_id"].split("_")[1]) for r in existing_runs if r["run_id"].startswith("run_")}
#     run_index = 0
#     while run_index in existing_run_ids:
#         run_index += 1
        
#     current_run_id = f"run_{run_index}"
#     target_run_folder_path = os.path.join(dataset_augmented_base_path, current_run_id)
#     os.makedirs(target_run_folder_path) # Already checked it doesn't exist effectively

#     zip_filename = f"{dataset_name}_augmented_{current_run_id}.zip"
#     zip_filepath = os.path.join(dataset_augmented_base_path, zip_filename) # Zip stored alongside run folders

#     source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     augmented_file_names_for_zip = []

#     for filename in source_image_files:
#         img_path = os.path.join(source_folder, filename)
#         try:
#             with Image.open(img_path) as img:
#                 augmented_img = run_augmentations(
#                     img, techniques, parameters,
#                     source_folder, source_image_files, filename
#                 )
#                 # Ensure image is in a savable mode (e.g., RGB/RGBA)
#                 if augmented_img.mode == 'P' and 'transparency' in augmented_img.info: 
#                     augmented_img = augmented_img.convert("RGBA")
#                 elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): 
#                     augmented_img = augmented_img.convert("RGBA")
#                 elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: 
#                     augmented_img = augmented_img.convert('RGB')

#                 base, ext = os.path.splitext(filename)
#                 # Use original extension if it's common, otherwise default to PNG
#                 save_ext = ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'
#                 save_filename = f"aug_{base}{save_ext}"
#                 save_path = os.path.join(target_run_folder_path, save_filename)

#                 if save_ext.lower() in ['.jpg', '.jpeg']: 
#                     # For JPEG, ensure image is RGB as 'RGBA' causes issues
#                     if augmented_img.mode == 'RGBA':
#                         rgb_img = Image.new("RGB", augmented_img.size, (255, 255, 255))
#                         rgb_img.paste(augmented_img, mask=augmented_img.split()[3]) # 3 is the alpha channel
#                         augmented_img = rgb_img
#                     augmented_img.save(save_path, "JPEG", quality=95)
#                 else: # PNG or other (defaulted to PNG)
#                     augmented_img.save(save_path, "PNG")
#                 augmented_file_names_for_zip.append(save_filename)
#         except Exception as e: 
#             print(f"Error augmenting image {filename}: {e}")
#             # import traceback # For debugging
#             # traceback.print_exc()

#     # Create zip file from the files in target_run_folder_path
#     with zipfile.ZipFile(zip_filepath, 'w') as zipf:
#         for aug_file_name in augmented_file_names_for_zip:
#             file_in_run_folder_path = os.path.join(target_run_folder_path, aug_file_name)
#             if os.path.isfile(file_in_run_folder_path):
#                  zipf.write(file_in_run_folder_path, aug_file_name) # Write file with its name, not full path

#     # Update metadata
#     if dataset_name not in augmentation_metadata: 
#         augmentation_metadata[dataset_name] = {"augmentation_runs": []}
#     elif "augmentation_runs" not in augmentation_metadata[dataset_name]: 
#         augmentation_metadata[dataset_name]["augmentation_runs"] = []
    
#     new_run_info = {
#         "run_id": current_run_id,
#         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#         "techniques": techniques, "parameters": parameters,
#         "augmented_zip": zip_filename, # Filename of the zip, not the full path
#         "output_folder_name": current_run_id 
#     }
#     augmentation_metadata[dataset_name]["augmentation_runs"].append(new_run_info)
#     with open(METADATA_FILE, 'w') as f: json.dump(augmentation_metadata, f, indent=4)
#     return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'run_id': current_run_id})


# @app.route('/preview_augmentation', methods=['POST']) # For SINGLE technique preview
# def preview_augmentation():
#     data = request.get_json()
#     dataset_name = secure_filename(data.get('datasetName'))
#     image_filename = secure_filename(data.get('imageFilename'))
#     technique = data.get('technique') # single technique string
#     params = data.get('parameters', {})

#     if not all([dataset_name, image_filename, technique]):
#         return jsonify({"error": "Missing data for single technique preview"}), 400

#     original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
#     if not os.path.exists(original_image_path):
#         return jsonify({"error": "Sample image not found"}), 404

#     try:
#         with Image.open(original_image_path) as img:
#             preview_image = img.copy()

#             # Apply ONLY the single specified technique
#             if technique == "rotate": preview_image = rotate_image(preview_image, float(params.get("rotation_angle", 90)))
#             elif technique == "scale": preview_image = scale_image(preview_image, float(params.get("scaling_factor", 1.5)))
#             elif technique == "translate":
#                 x_offset = int(params.get("translation_x", 0))
#                 y_offset = int(params.get("translation_y", 0))
#                 preview_image = translate_image(preview_image, x_offset, y_offset)
#             elif technique == "flip_horizontal": preview_image = flip_horizontal(preview_image)
#             elif technique == "flip_vertical": preview_image = flip_vertical(preview_image)
#             elif technique == "crop":
#                 img_w, img_h = preview_image.size
#                 left = float(params.get("crop_left", 0))
#                 top = float(params.get("crop_top", 0))
#                 right = float(params.get("crop_right", img_w))
#                 bottom = float(params.get("crop_bottom", img_h))
#                 preview_image = crop_image(preview_image, left, top, right, bottom)
#             elif technique == "pad":
#                 padding = int(params.get("padding_size", 0))
#                 padding_color = params.get("padding_color", "#000000")
#                 preview_image = pad_image(preview_image, padding, padding_color)
#             elif technique == "brightness": preview_image = adjust_brightness(preview_image, float(params.get("brightness_factor", 1.0)))
#             elif technique == "contrast": preview_image = adjust_contrast(preview_image, float(params.get("contrast_factor", 1.0)))
#             elif technique == "saturation": preview_image = adjust_saturation(preview_image, float(params.get("saturation_factor", 1.0)))
#             elif technique == "grayscale": preview_image = convert_grayscale(preview_image)
#             elif technique == "gaussian_noise": preview_image = add_gaussian_noise(preview_image, var=float(params.get("gaussian_variance", 0.01)))
#             elif technique == "salt_pepper_noise": preview_image = add_salt_pepper_noise(preview_image, amount=float(params.get("sap_amount", 0.005)))
#             elif technique == "speckle_noise": preview_image = add_speckle_noise(preview_image)
#             elif technique == "motion_blur": preview_image = add_motion_blur(preview_image, size=int(params.get("motion_blur_size", 9)))
#             elif technique == "cutout": preview_image = apply_cutout(preview_image, int(params.get("cutout_size", 50)))
#             elif technique == "random_erasing": preview_image = apply_random_erasing(preview_image)
#             # Mixup/Cutmix are not suitable for single technique preview here by design
#             else:
#                 # If technique is unknown or not handled for single preview, return original copy
#                 pass 

#             # Convert to base64
#             if preview_image.mode == 'P' and 'transparency' in preview_image.info: preview_image = preview_image.convert("RGBA")
#             elif preview_image.mode == 'LA' or (preview_image.mode == 'L' and 'transparency' in preview_image.info): preview_image = preview_image.convert("RGBA")
#             elif preview_image.mode not in ['RGB', 'RGBA', 'L']: preview_image = preview_image.convert('RGB')
            
#             buffered = io.BytesIO()
#             # For JPEG, ensure image is RGB
#             save_format = "PNG" # Default to PNG for previews to support alpha
#             if preview_image.format == "JPEG" and preview_image.mode == 'RGBA':
#                  # If original was JPEG and became RGBA, save as PNG to keep alpha
#                  # Or convert to RGB if JPEG output is strictly needed
#                  pass # Keep as PNG for preview
            
#             preview_image.save(buffered, format=save_format) 
#             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#             return jsonify({"preview_image_base64": f"data:image/{save_format.lower()};base64,{img_str}"})

#     except Exception as e:
#         print(f"Error generating single technique preview for {technique}: {e}")
#         # import traceback
#         # traceback.print_exc()
#         return jsonify({"error": f"Could not generate single technique preview: {str(e)}"}), 500

# # NEW ENDPOINT FOR COMBINED PREVIEW
# @app.route('/preview_combined_augmentations', methods=['POST'])
# def preview_combined_augmentations():
#     data = request.get_json()
#     dataset_name = secure_filename(data.get('datasetName'))
#     image_filename = secure_filename(data.get('imageFilename'))
#     techniques_to_apply = data.get('techniques', []) # List of techniques
#     params = data.get('parameters', {})

#     if not all([dataset_name, image_filename]):
#         return jsonify({"error": "Missing dataset name or image filename for combined preview"}), 400
    
#     original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
#     source_dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)

#     if not os.path.exists(original_image_path):
#         return jsonify({"error": "Sample image not found for combined preview"}), 404
#     if not os.path.isdir(source_dataset_folder): # Needed for mixup/cutmix context
#          return jsonify({"error": f"Source dataset folder '{dataset_name}' not found for combined preview."}), 404

#     try:
#         files_in_dataset = []
#         if os.path.isdir(source_dataset_folder): # Ensure it is a directory
#             files_in_dataset = [f for f in os.listdir(source_dataset_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#         with Image.open(original_image_path) as img:
#             processed_image = run_augmentations(
#                 img,
#                 techniques_to_apply,
#                 params,
#                 source_dataset_folder,
#                 files_in_dataset,
#                 image_filename 
#             )

#             if processed_image.mode == 'P' and 'transparency' in processed_image.info: processed_image = processed_image.convert("RGBA")
#             elif processed_image.mode == 'LA' or (processed_image.mode == 'L' and 'transparency' in processed_image.info): processed_image = processed_image.convert("RGBA")
#             elif processed_image.mode not in ['RGB', 'RGBA', 'L']: processed_image = processed_image.convert('RGB')

#             buffered = io.BytesIO()
#             save_format="PNG" # PNG for previews
#             processed_image.save(buffered, format=save_format)
#             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#             return jsonify({"preview_image_base64": f"data:image/{save_format.lower()};base64,{img_str}"})

#     except Exception as e:
#         print(f"Error generating combined augmentations preview: {e}")
#         # import traceback
#         # traceback.print_exc()
#         return jsonify({"error": f"Could not generate combined preview: {str(e)}"}), 500


# @app.route('/uploads/<dataset>/<filename>')
# def uploaded_file(dataset, filename):
#     return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

# @app.route('/augmented/<dataset>/<zipfilename>') 
# def serve_augmented_zip(dataset, zipfilename):
#     # zip files are stored directly under AUGMENTED_FOLDER/dataset_name/
#     dataset_aug_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset))
#     return send_from_directory(dataset_aug_path, secure_filename(zipfilename), as_attachment=True)

# @app.route('/augmented_image/<dataset>/<run_id>/<filename>')
# def serve_augmented_image(dataset, run_id, filename):
#     # Individual augmented images are inside AUGMENTED_FOLDER/dataset_name/run_id/
#     image_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_id))
#     return send_from_directory(image_path, secure_filename(filename))


# @app.route('/generate-synthetic', methods=['POST'])
# def generate_synthetic():
#     file = request.files.get('file')
#     epochs = int(request.form.get('epochs', 5)) # Default to 5 if not provided
#     num_samples = int(request.form.get('samples', 100)) # Default to 100 if not provided

#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     if not GRADIO_URL:
#         return jsonify({"error": "Gradio service URL is not configured on the server."}), 500

#     # Save temp uploaded file
#     temp_input_filename = secure_filename(file.filename) # Secure the filename
#     temp_input_path = os.path.join(csv_DIR, temp_input_filename)
#     file.save(temp_input_path)

#     try:
#         client = Client(GRADIO_URL)
#         # The predict function might return a file path string or a FileData object
#         # depending on the Gradio version and component. Assuming it's a path.
#         prediction_result_path = client.predict(
#             data=handle_file(temp_input_path), # Input CSV file path
#             epochs=epochs,                     # Number of epochs
#             num_samples=num_samples,           # Number of samples to generate
#             api_name="/predict"                # Or whatever the API endpoint name is in your Gradio app
#         )
        
#         # Ensure prediction_result_path is a string (path to the generated file)
#         if not isinstance(prediction_result_path, str) or not os.path.exists(prediction_result_path):
#              print(f"Gradio client did not return a valid file path. Result: {prediction_result_path}")
#              return jsonify({"error": "Synthetic data generation failed or returned unexpected result."}), 500

#         # Create a filename for the output in RESULT_DIR
#         base, ext = os.path.splitext(temp_input_filename)
#         output_filename = f"{base}_gen_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
#         output_path = os.path.join(RESULT_DIR, output_filename)
        
#         shutil.copy(prediction_result_path, output_path)

#         # Clean up the temporary input file from csv_DIR
#         if os.path.exists(temp_input_path):
#             os.remove(temp_input_path)
#         # Clean up the file Gradio client might have saved (if it's in a temp location different from output_path)
#         # This depends on how Gradio client handles output files. If prediction_result_path is temporary,
#         # and shutil.copy was successful, it might be safe to remove it.
#         # However, without knowing the exact behavior, it's safer to let Gradio or system temp cleanup handle it.
#         # If prediction_result_path points to a file inside a temp directory managed by the gradio_client,
#         # it should be cleaned up automatically when the client or its temp files are handled.

#         return jsonify({
#             "message": "Synthetic data generated successfully.",
#             "output_file": output_filename # Send the new filename for download
#         })

#     except Exception as e:
#         print(f"Error during synthetic data generation: {e}")
#         # import traceback
#         # traceback.print_exc()
#         # Clean up the temp input file in case of error too
#         if os.path.exists(temp_input_path):
#             os.remove(temp_input_path)
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# @app.route('/download-csv/<filename>', methods=['GET'])
# def download_file(filename):
#     # Secure the filename again before joining path
#     safe_filename = secure_filename(filename)
#     if not safe_filename == filename: # check if secure_filename changed the input
#         return jsonify({"error": "Invalid filename"}), 400
        
#     filepath = os.path.join(RESULT_DIR, safe_filename)
#     if not os.path.isfile(filepath):
#         return jsonify({"error": "File not found or access denied"}), 404
        
#     return send_file(filepath, as_attachment=True)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)




import os
import zipfile
import json
import random
import numpy as np
import cv2 # For motion blur
import datetime # For timestamps
import io # For image byte streaming
import base64 # For base64 encoding

from flask import Flask, request, jsonify, send_from_directory,send_file

from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageChops

# import os # Already imported
# from flask import Flask, request, jsonify # Already imported
from gradio_client import Client, handle_file
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

GRADIO_URL = os.getenv("GRADIO_URL")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
AUGMENTED_FOLDER = 'augmented'
METADATA_FILE = 'augmentation_metadata.json'
RESULT_DIR = "csv_results"
csv_DIR = "csv_uploads"

# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(csv_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        augmentation_metadata = json.load(f)
else:
    augmentation_metadata = {}

# --- Start of Augmentation Functions ---
def rotate_image(image, angle=90):
    return image.rotate(angle, expand=True)

def scale_image(image, scale=1.5):
    if scale <= 0: return image
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)))

def translate_image(image, x_offset, y_offset):
    return ImageChops.offset(image, x_offset, y_offset)

def flip_horizontal(image):
    return ImageOps.mirror(image)

def flip_vertical(image):
    return ImageOps.flip(image)

def crop_image(image, left, top, right, bottom):
    width, height = image.size
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(right))
    bottom = min(height, int(bottom))
    if left < right and top < bottom:
        return image.crop((left, top, right, bottom))
    return image

def pad_image(image, padding_size, padding_color="#000000"):
    padding_size = int(padding_size)
    if padding_size <=0: return image
    return ImageOps.expand(image, border=padding_size, fill=padding_color)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image.convert('RGB'))
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image.convert('RGB'))
    return enhancer.enhance(factor)

def convert_grayscale(image):
    return ImageOps.grayscale(image).convert('RGB')

def adjust_saturation(image, factor):
    enhancer = ImageEnhance.Color(image.convert('RGB'))
    return enhancer.enhance(factor)

def add_gaussian_noise(image, mean=0, var=0.01):
    img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    noise = np.random.normal(mean, var ** 0.5, img_array.shape)
    img_noisy = img_array + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    img_array = np.array(image.convert('RGB'))
    num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)
    if img_array.size > 0 :
        if num_salt > 0:
            salt_rows = np.random.randint(0, max(1, img_array.shape[0]), num_salt)
            salt_cols = np.random.randint(0, max(1, img_array.shape[1]), num_salt)
            img_array[salt_rows, salt_cols, :] = [255,255,255]
        if num_pepper > 0:
            pepper_rows = np.random.randint(0, max(1, img_array.shape[0]), num_pepper)
            pepper_cols = np.random.randint(0, max(1, img_array.shape[1]), num_pepper)
            img_array[pepper_rows, pepper_cols, :] = [0,0,0]
    return Image.fromarray(img_array)

def add_speckle_noise(image):
    img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    noise = np.random.randn(*img_array.shape)
    img_noisy = img_array + img_array * noise
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def add_motion_blur(image, size=9):
    size = int(size)
    if size <= 1: size = 3
    if size % 2 == 0 : size +=1
    kernel = np.zeros((size, size))
    kernel[int((size - 1)/2), :] = np.ones(size)
    kernel = kernel / size
    img_array = np.array(image.convert('RGB'))
    img_blur = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(img_blur)

def apply_cutout(image, mask_size):
    mask_size = int(mask_size)
    if mask_size <=0: return image
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]
    if h == 0 or w == 0: return image
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    img_array[y1:y2, x1:x2] = 0
    return Image.fromarray(img_array)

def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
    img_array = np.array(image.convert('RGB'))
    h, w, c = img_array.shape
    s_img = h * w
    if s_img == 0: return image
    s_erase = np.random.uniform(sl, sh) * s_img
    r_aspect = np.random.uniform(r1, 1/r1 if r1 != 0 else 1)
    h_e = int(np.sqrt(s_erase * r_aspect))
    w_e = int(np.sqrt(s_erase / r_aspect if r_aspect != 0 else s_erase))
    if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h:
        return Image.fromarray(img_array)
    x_e = np.random.randint(0, w - w_e + 1)
    y_e = np.random.randint(0, h - h_e + 1)
    img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
    return Image.fromarray(img_array)

def apply_mixup(image, other_image, alpha=0.4):
    if other_image is None: return image
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    image_array = np.array(image.convert('RGB')).astype(np.float32)
    other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
    mixed_array = lam * image_array + (1 - lam) * other_array
    return Image.fromarray(mixed_array.astype(np.uint8))

def apply_cutmix(image, other_image):
    if other_image is None: return image
    img_array = np.array(image.convert('RGB'))
    other_array = np.array(other_image.convert('RGB').resize(image.size))
    h, w, _ = img_array.shape
    if h == 0 or w == 0: return image
    lam = np.random.beta(1.0, 1.0)
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    if cut_w == 0 or cut_h == 0: return image
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    if bbx1 < bbx2 and bby1 < bby2:
      img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
    return Image.fromarray(img_array)

def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
    processed_image = image.copy()
    if "rotate" in techniques:
        processed_image = rotate_image(processed_image, float(params.get("rotation_angle", 90)))
    if "scale" in techniques:
        processed_image = scale_image(processed_image, float(params.get("scaling_factor", 1.5)))
    if "translate" in techniques:
        processed_image = translate_image(processed_image, int(params.get("translation_x", 0)), int(params.get("translation_y", 0)))
    if "flip_horizontal" in techniques:
        processed_image = flip_horizontal(processed_image)
    if "flip_vertical" in techniques:
        processed_image = flip_vertical(processed_image)
    if "crop" in techniques:
        img_w, img_h = processed_image.size
        processed_image = crop_image(processed_image, float(params.get("crop_left", 0)), float(params.get("crop_top", 0)), float(params.get("crop_right", img_w)), float(params.get("crop_bottom", img_h)))
    if "pad" in techniques:
        processed_image = pad_image(processed_image, int(params.get("padding_size", 0)), params.get("padding_color", "#000000"))
    if "brightness" in techniques:
        processed_image = adjust_brightness(processed_image, float(params.get("brightness_factor", 1.0)))
    if "contrast" in techniques:
        processed_image = adjust_contrast(processed_image, float(params.get("contrast_factor", 1.0)))
    if "grayscale" in techniques:
        processed_image = convert_grayscale(processed_image)
    if "saturation" in techniques:
        processed_image = adjust_saturation(processed_image, float(params.get("saturation_factor", 1.0)))
    if "gaussian_noise" in techniques:
        processed_image = add_gaussian_noise(processed_image, var=float(params.get("gaussian_variance", 0.01)))
    if "salt_pepper_noise" in techniques:
        processed_image = add_salt_pepper_noise(processed_image, amount=float(params.get("sap_amount", 0.005)))
    if "speckle_noise" in techniques:
        processed_image = add_speckle_noise(processed_image)
    if "motion_blur" in techniques:
        processed_image = add_motion_blur(processed_image, size=int(params.get("motion_blur_size", 9)))
    if "cutout" in techniques:
        processed_image = apply_cutout(processed_image, int(params.get("cutout_size", 50)))
    if "random_erasing" in techniques:
        processed_image = apply_random_erasing(processed_image)
    
    other_image_for_mix = None
    if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset and source_dataset_folder:
        possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
        if not possible_other_files and len(files_in_dataset) > 0:
            possible_other_files = files_in_dataset
        if possible_other_files:
            other_filename = random.choice(possible_other_files)
            other_filepath = os.path.join(source_dataset_folder, other_filename)
            try:
                other_image_for_mix = Image.open(other_filepath)
            except Exception as e:
                print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
    if "mixup" in techniques:
        processed_image = apply_mixup(processed_image, other_image_for_mix, float(params.get("mixup_alpha", 0.4)))
    if "cutmix" in techniques:
        processed_image = apply_cutmix(processed_image, other_image_for_mix)
    if other_image_for_mix:
        other_image_for_mix.close()
    return processed_image
# --- End of Augmentation Functions ---

@app.route('/upload', methods=['POST'])
def upload_files():
    dataset = request.form.get('dataset')
    files = request.files.getlist('files')
    if not dataset:
        return jsonify({"error": "No dataset name provided"}), 400
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected for upload"}), 400
    dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
    os.makedirs(dataset_folder, exist_ok=True)
    for file in files:
        if file.filename == '': continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(dataset_folder, filename)
        file.save(filepath)
        if filename.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_folder)
                os.remove(filepath)
            except zipfile.BadZipFile:
                print(f"Bad zip file: {filepath}. Kept for inspection.")
            except Exception as e:
                print(f"Error processing zip file {filepath}: {e}")
    return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200

@app.route('/datasets', methods=['GET'])
def list_datasets():
    datasets_info = []
    base_upload_folder = app.config['UPLOAD_FOLDER']
    for folder_name in os.listdir(base_upload_folder):
        folder_path = os.path.join(base_upload_folder, folder_name)
        if os.path.isdir(folder_path):
            try:
                files_in_folder = os.listdir(folder_path)
                image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                dataset_metadata = augmentation_metadata.get(folder_name, {})
                runs = dataset_metadata.get("augmentation_runs", [])
                datasets_info.append({
                    'name': folder_name,
                    'count': len(image_files),
                    'files': image_files,
                    'all_files_in_folder': files_in_folder,
                    'augmentation_runs': runs
                })
            except Exception as e:
                print(f"Error listing dataset {folder_name}: {e}")
    return jsonify(datasets_info)

@app.route('/augment', methods=['POST'])
def augment_dataset_route():
    data = request.get_json()
    dataset_name = secure_filename(data.get('datasetName'))
    techniques = data.get('techniques', [])
    parameters = data.get('parameters', {})
    if not dataset_name: return jsonify({'error': 'Dataset name not provided'}), 400
    if not techniques: return jsonify({'error': 'No augmentation techniques selected'}), 400
    source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    if not os.path.isdir(source_folder):
        return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404
    dataset_augmented_base_path = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
    os.makedirs(dataset_augmented_base_path, exist_ok=True)
    existing_runs = augmentation_metadata.get(dataset_name, {}).get("augmentation_runs", [])
    existing_run_ids = {int(r["run_id"].split("_")[1]) for r in existing_runs if r["run_id"].startswith("run_") and r["run_id"].split("_")[1].isdigit()}
    run_index = 0
    while run_index in existing_run_ids:
        run_index += 1
    current_run_id = f"run_{run_index}"
    target_run_folder_path = os.path.join(dataset_augmented_base_path, current_run_id)
    os.makedirs(target_run_folder_path)
    zip_filename = f"{dataset_name}_augmented_{current_run_id}.zip"
    zip_filepath = os.path.join(dataset_augmented_base_path, zip_filename)
    source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    augmented_file_names_for_zip = []
    for filename in source_image_files:
        img_path = os.path.join(source_folder, filename)
        try:
            with Image.open(img_path) as img:
                augmented_img = run_augmentations(img, techniques, parameters, source_folder, source_image_files, filename)
                if augmented_img.mode == 'P' and 'transparency' in augmented_img.info: augmented_img = augmented_img.convert("RGBA")
                elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): augmented_img = augmented_img.convert("RGBA")
                elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: augmented_img = augmented_img.convert('RGB')
                base, ext = os.path.splitext(filename)
                save_ext = ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'
                save_filename = f"aug_{base}{save_ext}"
                save_path = os.path.join(target_run_folder_path, save_filename)
                if save_ext.lower() in ['.jpg', '.jpeg']:
                    if augmented_img.mode == 'RGBA':
                        rgb_img = Image.new("RGB", augmented_img.size, (255, 255, 255))
                        rgb_img.paste(augmented_img, mask=augmented_img.split()[3])
                        augmented_img = rgb_img
                    augmented_img.save(save_path, "JPEG", quality=95)
                else:
                    augmented_img.save(save_path, "PNG")
                augmented_file_names_for_zip.append(save_filename)
        except Exception as e: print(f"Error augmenting image {filename}: {e}")
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for aug_file_name in augmented_file_names_for_zip:
            file_in_run_folder_path = os.path.join(target_run_folder_path, aug_file_name)
            if os.path.isfile(file_in_run_folder_path):
                 zipf.write(file_in_run_folder_path, aug_file_name)
    if dataset_name not in augmentation_metadata: augmentation_metadata[dataset_name] = {"augmentation_runs": []}
    elif "augmentation_runs" not in augmentation_metadata[dataset_name]: augmentation_metadata[dataset_name]["augmentation_runs"] = []
    new_run_info = {
        "run_id": current_run_id, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "techniques": techniques, "parameters": parameters, "augmented_zip": zip_filename,
        "output_folder_name": current_run_id
    }
    augmentation_metadata[dataset_name]["augmentation_runs"].append(new_run_info)
    with open(METADATA_FILE, 'w') as f: json.dump(augmentation_metadata, f, indent=4)
    return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'run_id': current_run_id})

@app.route('/preview_augmentation', methods=['POST'])
def preview_augmentation():
    data = request.get_json()
    dataset_name = secure_filename(data.get('datasetName'))
    image_filename = secure_filename(data.get('imageFilename'))
    technique = data.get('technique')
    params = data.get('parameters', {})
    if not all([dataset_name, image_filename, technique]):
        return jsonify({"error": "Missing data for single technique preview"}), 400
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
    if not os.path.exists(original_image_path):
        return jsonify({"error": "Sample image not found"}), 404
    try:
        with Image.open(original_image_path) as img:
            preview_image = img.copy()
            if technique == "rotate": preview_image = rotate_image(preview_image, float(params.get("rotation_angle", 90)))
            elif technique == "scale": preview_image = scale_image(preview_image, float(params.get("scaling_factor", 1.5)))
            elif technique == "translate": preview_image = translate_image(preview_image, int(params.get("translation_x", 0)), int(params.get("translation_y", 0)))
            elif technique == "flip_horizontal": preview_image = flip_horizontal(preview_image)
            elif technique == "flip_vertical": preview_image = flip_vertical(preview_image)
            elif technique == "crop":
                img_w, img_h = preview_image.size
                preview_image = crop_image(preview_image, float(params.get("crop_left", 0)), float(params.get("crop_top", 0)), float(params.get("crop_right", img_w)), float(params.get("crop_bottom", img_h)))
            elif technique == "pad": preview_image = pad_image(preview_image, int(params.get("padding_size", 0)), params.get("padding_color", "#000000"))
            elif technique == "brightness": preview_image = adjust_brightness(preview_image, float(params.get("brightness_factor", 1.0)))
            elif technique == "contrast": preview_image = adjust_contrast(preview_image, float(params.get("contrast_factor", 1.0)))
            elif technique == "saturation": preview_image = adjust_saturation(preview_image, float(params.get("saturation_factor", 1.0)))
            elif technique == "grayscale": preview_image = convert_grayscale(preview_image)
            elif technique == "gaussian_noise": preview_image = add_gaussian_noise(preview_image, var=float(params.get("gaussian_variance", 0.01)))
            elif technique == "salt_pepper_noise": preview_image = add_salt_pepper_noise(preview_image, amount=float(params.get("sap_amount", 0.005)))
            elif technique == "speckle_noise": preview_image = add_speckle_noise(preview_image)
            elif technique == "motion_blur": preview_image = add_motion_blur(preview_image, size=int(params.get("motion_blur_size", 9)))
            elif technique == "cutout": preview_image = apply_cutout(preview_image, int(params.get("cutout_size", 50)))
            elif technique == "random_erasing": preview_image = apply_random_erasing(preview_image)
            if preview_image.mode == 'P' and 'transparency' in preview_image.info: preview_image = preview_image.convert("RGBA")
            elif preview_image.mode == 'LA' or (preview_image.mode == 'L' and 'transparency' in preview_image.info): preview_image = preview_image.convert("RGBA")
            elif preview_image.mode not in ['RGB', 'RGBA', 'L']: preview_image = preview_image.convert('RGB')
            buffered = io.BytesIO()
            save_format = "PNG"
            preview_image.save(buffered, format=save_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return jsonify({"preview_image_base64": f"data:image/{save_format.lower()};base64,{img_str}"})
    except Exception as e:
        print(f"Error generating single technique preview for {technique}: {e}")
        return jsonify({"error": f"Could not generate single technique preview: {str(e)}"}), 500

@app.route('/preview_combined_augmentations', methods=['POST'])
def preview_combined_augmentations():
    data = request.get_json()
    dataset_name = secure_filename(data.get('datasetName'))
    image_filename = secure_filename(data.get('imageFilename'))
    techniques_to_apply = data.get('techniques', [])
    params = data.get('parameters', {})
    if not all([dataset_name, image_filename]):
        return jsonify({"error": "Missing dataset name or image filename for combined preview"}), 400
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name, image_filename)
    source_dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    if not os.path.exists(original_image_path):
        return jsonify({"error": "Sample image not found for combined preview"}), 404
    if not os.path.isdir(source_dataset_folder):
         return jsonify({"error": f"Source dataset folder '{dataset_name}' not found for combined preview."}), 404
    try:
        files_in_dataset = []
        if os.path.isdir(source_dataset_folder):
            files_in_dataset = [f for f in os.listdir(source_dataset_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with Image.open(original_image_path) as img:
            processed_image = run_augmentations(img, techniques_to_apply, params, source_dataset_folder, files_in_dataset, image_filename)
            if processed_image.mode == 'P' and 'transparency' in processed_image.info: processed_image = processed_image.convert("RGBA")
            elif processed_image.mode == 'LA' or (processed_image.mode == 'L' and 'transparency' in processed_image.info): processed_image = processed_image.convert("RGBA")
            elif processed_image.mode not in ['RGB', 'RGBA', 'L']: processed_image = processed_image.convert('RGB')
            buffered = io.BytesIO()
            save_format="PNG"
            processed_image.save(buffered, format=save_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return jsonify({"preview_image_base64": f"data:image/{save_format.lower()};base64,{img_str}"})
    except Exception as e:
        print(f"Error generating combined augmentations preview: {e}")
        return jsonify({"error": f"Could not generate combined preview: {str(e)}"}), 500

@app.route('/uploads/<dataset>/<filename>')
def uploaded_file(dataset, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

@app.route('/augmented/<dataset>/<zipfilename>')
def serve_augmented_zip(dataset, zipfilename):
    dataset_aug_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset))
    return send_from_directory(dataset_aug_path, secure_filename(zipfilename), as_attachment=True)

@app.route('/augmented_image/<dataset>/<run_id>/<filename>')
def serve_augmented_image(dataset, run_id, filename):
    image_path = os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_id))
    return send_from_directory(image_path, secure_filename(filename))

@app.route('/generate-synthetic', methods=['POST'])
def generate_synthetic():
    file = request.files.get('file')
    epochs = int(request.form.get('epochs', 5))
    num_samples = int(request.form.get('samples', 100))
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if not GRADIO_URL:
        return jsonify({"error": "Gradio service URL is not configured on the server."}), 500
    temp_input_filename = secure_filename(file.filename)
    temp_input_path = os.path.join(csv_DIR, temp_input_filename)
    file.save(temp_input_path)
    try:
        client = Client(GRADIO_URL)
        prediction_result_path = client.predict(
            data=handle_file(temp_input_path),
            epochs=epochs,
            num_samples=num_samples,
            api_name="/predict"
        )
        if not isinstance(prediction_result_path, str) or not os.path.exists(prediction_result_path):
             print(f"Gradio client did not return a valid file path. Result: {prediction_result_path}")
             return jsonify({"error": "Synthetic data generation failed or returned unexpected result."}), 500
        base, ext = os.path.splitext(temp_input_filename)
        output_filename = f"{base}_gen_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        output_path = os.path.join(RESULT_DIR, output_filename)
        shutil.copy(prediction_result_path, output_path)
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"message": "Synthetic data generated successfully.", "output_file": output_filename})
    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/download-csv/<filename>', methods=['GET'])
def download_file(filename):
    safe_filename = secure_filename(filename)
    if not safe_filename == filename:
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(RESULT_DIR, safe_filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found or access denied"}), 404
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)