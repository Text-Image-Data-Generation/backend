# # app.py
# import os
# import zipfile
# import json
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from PIL import Image, ImageOps

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = 'uploads'
# AUGMENTED_FOLDER = 'augmented'
# METADATA_FILE = 'augmentation_metadata.json'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# # Load augmentation metadata
# if os.path.exists(METADATA_FILE):
#     with open(METADATA_FILE, 'r') as f:
#         augmentation_metadata = json.load(f)
# else:
#     augmentation_metadata = {}

# # Augmentation functions
# def rotate_image(image, angle=90):
#     return image.rotate(angle, expand=True)

# def scale_image(image, scale=1.5):
#     width, height = image.size
#     return image.resize((int(width * scale), int(height * scale)))

# def flip_horizontal(image):
#     return ImageOps.mirror(image)

# def flip_vertical(image):
#     return ImageOps.flip(image)

# def apply_augmentations(image, augmentations):
#     for aug in augmentations:
#         if aug == 'rotate':
#             image = rotate_image(image)
#         elif aug == 'scale':
#             image = scale_image(image)
#         elif aug == 'flip_horizontal':
#             image = flip_horizontal(image)
#         elif aug == 'flip_vertical':
#             image = flip_vertical(image)
#     return image

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     dataset = request.form.get('dataset')
#     files = request.files.getlist('files')

#     if not dataset:
#         return "No dataset name provided", 400

#     dataset_folder = os.path.join(UPLOAD_FOLDER, secure_filename(dataset))
#     os.makedirs(dataset_folder, exist_ok=True)

#     for file in files:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(dataset_folder, filename)
#         file.save(filepath)

#         if filename.endswith('.zip'):
#             with zipfile.ZipFile(filepath, 'r') as zip_ref:
#                 zip_ref.extractall(dataset_folder)
#             os.remove(filepath)

#     return "Files uploaded", 200

# @app.route('/datasets', methods=['GET'])
# def list_datasets():
#     datasets = []
#     for folder in os.listdir(UPLOAD_FOLDER):
#         folder_path = os.path.join(UPLOAD_FOLDER, folder)
#         if os.path.isdir(folder_path):
#             files = os.listdir(folder_path)
#             metadata = augmentation_metadata.get(folder, {})
#             datasets.append({
#                 'name': folder,
#                 'count': len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]),
#                 'files': files,
#                 'augmentations': metadata.get("augmentations", []),
#                 'augmented_zip': metadata.get("augmented_zip")
#             })
#     return jsonify(datasets)

# @app.route('/augment', methods=['POST'])
# def augment_dataset():
#     data = request.get_json()
#     dataset_name = secure_filename(data.get('datasetName'))
#     augmentations = data.get('augmentations', [])

#     source_folder = os.path.join(UPLOAD_FOLDER, dataset_name)
#     target_folder = os.path.join(AUGMENTED_FOLDER, dataset_name)
#     os.makedirs(target_folder, exist_ok=True)

#     for filename in os.listdir(source_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(source_folder, filename)
#             with Image.open(img_path) as img:
#                 augmented_img = apply_augmentations(img, augmentations)
#                 save_path = os.path.join(target_folder, f"aug_{filename}")
#                 augmented_img.save(save_path)

#     # Zip the augmented folder
#     zip_filename = f"{dataset_name}_augmented.zip"
#     zip_path = os.path.join(target_folder, zip_filename)
#     with zipfile.ZipFile(zip_path, 'w') as zipf:
#         for file in os.listdir(target_folder):
#             if file.endswith(('.png', '.jpg', '.jpeg')):
#                 zipf.write(os.path.join(target_folder, file), file)

#     # Save metadata
#     augmentation_metadata[dataset_name] = {
#         "augmentations": augmentations,
#         "augmented_zip": zip_filename
#     }
#     with open(METADATA_FILE, 'w') as f:
#         json.dump(augmentation_metadata, f)

#     return jsonify({'message': 'Augmentation complete', 'zip': zip_filename})

# @app.route('/uploads/<dataset>/<filename>')
# def uploaded_file(dataset, filename):
#     return send_from_directory(os.path.join(UPLOAD_FOLDER, secure_filename(dataset)), filename)

# @app.route('/augmented/<dataset>/<filename>')
# def augmented_file(dataset, filename):
#     return send_from_directory(os.path.join(AUGMENTED_FOLDER, secure_filename(dataset)), filename)

# from flask import send_from_directory

# @app.route('/download/<dataset_name>/<filename>')
# def download_file(dataset_name, filename):
#     augmented_dir = os.path.join('augmented', dataset_name)
#     return send_from_directory(augmented_dir, filename, as_attachment=True)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)





import os
import zipfile
import json
from flask import Flask, request, jsonify, send_from_directory , send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageChops

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
AUGMENTED_FOLDER = 'augmented'
METADATA_FILE = 'augmentation_metadata.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# Load augmentation metadata
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        augmentation_metadata = json.load(f)
else:
    augmentation_metadata = {}

# --- Start of Augmentation Functions ---

# Geometric Transformations
def rotate_image(image, angle=90):
    return image.rotate(angle, expand=True)

def scale_image(image, scale=1.5):
    if scale <= 0: return image # Avoid invalid scale
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
    # Ensure crop box is within image dimensions and valid
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(right))
    bottom = min(height, int(bottom))
    if left < right and top < bottom:
        return image.crop((left, top, right, bottom))
    return image # Return original if crop dimensions are invalid


def pad_image(image, padding_size, padding_color="#000000"):
    padding_size = int(padding_size)
    if padding_size <=0: return image
    return ImageOps.expand(image, border=padding_size, fill=padding_color)

# Color Transformations
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image.convert('RGB')) # Ensure RGB for brightness
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image.convert('RGB')) # Ensure RGB for contrast
    return enhancer.enhance(factor)

def convert_grayscale(image):
    return ImageOps.grayscale(image).convert('RGB') # Often want to keep 3 channels for consistency

def adjust_saturation(image, factor):
    enhancer = ImageEnhance.Color(image.convert('RGB')) # Ensure RGB for saturation
    return enhancer.enhance(factor)

# Noise Transformations
def add_gaussian_noise(image, mean=0, var=0.01):
    img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    noise = np.random.normal(mean, var ** 0.5, img_array.shape)
    img_noisy = img_array + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def add_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    img_array = np.array(image.convert('RGB')) # Work with RGB
    num_salt = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * img_array.shape[0] * img_array.shape[1] * (1.0 - salt_vs_pepper)).astype(int)

    # Salt noise
    coords_salt = [np.random.randint(0, i - 1 if i > 1 else 1, num_salt) for i in img_array.shape[:2]]
    if num_salt > 0 : img_array[coords_salt[0], coords_salt[1], :] = [255,255,255]


    # Pepper noise
    coords_pepper = [np.random.randint(0, i - 1 if i > 1 else 1, num_pepper) for i in img_array.shape[:2]]
    if num_pepper > 0: img_array[coords_pepper[0], coords_pepper[1], :] = [0,0,0]

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
    if size <= 0 : return image
    kernel = np.zeros((size, size))
    kernel[int((size - 1)/2), :] = np.ones(size)
    kernel = kernel / size
    img_array = np.array(image.convert('RGB'))
    img_blur = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(img_blur)

# Occlusion Transformations
def apply_cutout(image, mask_size):
    mask_size = int(mask_size)
    if mask_size <=0: return image
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]
    if h == 0 or w == 0: return image # Empty image

    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    img_array[y1:y2, x1:x2] = 0  # Black box
    return Image.fromarray(img_array)

def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
    img_array = np.array(image.convert('RGB'))
    h, w, c = img_array.shape # expect 3 channels for random color fill

    s_img = h * w
    s_erase = np.random.uniform(sl, sh) * s_img
    r_aspect = np.random.uniform(r1, 1/r1)

    h_e = int(np.sqrt(s_erase * r_aspect))
    w_e = int(np.sqrt(s_erase / r_aspect))

    if w_e == 0 or h_e == 0 or w_e >= w or h_e >= h: # check if erase dimensions are valid
        return Image.fromarray(img_array)

    x_e = np.random.randint(0, w - w_e + 1) # +1 for upper bound
    y_e = np.random.randint(0, h - h_e + 1)

    img_array[y_e:y_e+h_e, x_e:x_e+w_e] = np.random.randint(0, 256, (h_e, w_e, c))
    return Image.fromarray(img_array)

# Mix Transformations
def apply_mixup(image, other_image, alpha=0.4):
    if other_image is None: return image
    lam = np.random.beta(alpha, alpha)
    image_array = np.array(image.convert('RGB')).astype(np.float32)
    # Ensure other_image is RGB and same size
    other_array = np.array(other_image.convert('RGB').resize(image.size)).astype(np.float32)
    mixed_array = lam * image_array + (1 - lam) * other_array
    return Image.fromarray(mixed_array.astype(np.uint8))

def apply_cutmix(image, other_image):
    if other_image is None: return image
    img_array = np.array(image.convert('RGB'))
    # Ensure other_image is RGB and same size
    other_array = np.array(other_image.convert('RGB').resize(image.size))
    
    h, w, _ = img_array.shape
    if h == 0 or w == 0: return image

    lam = np.random.beta(1.0, 1.0) # typically 1.0, 1.0 for cutmix
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    if cut_w == 0 or cut_h == 0: return image # no actual cut

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    if bbx1 < bbx2 and bby1 < bby2: # Ensure valid box
      img_array[bby1:bby2, bbx1:bbx2, :] = other_array[bby1:bby2, bbx1:bbx2, :]
    return Image.fromarray(img_array)


# Master Augmentation Application Function
def run_augmentations(image, techniques, params, source_dataset_folder, files_in_dataset, current_image_filename):
    """Apply a series of augmentations based on selected techniques and parameters."""
    # Geometric Transformations
    if "rotate" in techniques: # Renamed from "rotation" for consistency
        angle = float(params.get("rotation_angle", 90))
        image = rotate_image(image, angle)
    if "scale" in techniques: # Renamed from "scaling"
        scale_factor = float(params.get("scaling_factor", 1.5))
        image = scale_image(image, scale_factor)
    if "translate" in techniques: # Renamed from "translation"
        x_offset = int(params.get("translation_x", 0))
        y_offset = int(params.get("translation_y", 0))
        image = translate_image(image, x_offset, y_offset)
    if "flip_horizontal" in techniques: # Renamed
        image = flip_horizontal(image)
    if "flip_vertical" in techniques: # Renamed
        image = flip_vertical(image)
    if "crop" in techniques: # Renamed
        left = int(params.get("crop_left", 0))
        top = int(params.get("crop_top", 0))
        right = int(params.get("crop_right", image.width if image else 0)) # Default to image width
        bottom = int(params.get("crop_bottom", image.height if image else 0)) # Default to image height
        image = crop_image(image, left, top, right, bottom)
    if "pad" in techniques: # Renamed
        padding = int(params.get("padding_size", 0))
        padding_color = params.get("padding_color", "#000000")
        image = pad_image(image, padding, padding_color)

    # Color Transformations
    if "brightness" in techniques:
        image = adjust_brightness(image, float(params.get("brightness_factor", 1.0)))
    if "contrast" in techniques:
        image = adjust_contrast(image, float(params.get("contrast_factor", 1.0)))
    if "grayscale" in techniques:
        image = convert_grayscale(image)
    if "saturation" in techniques:
        image = adjust_saturation(image, float(params.get("saturation_factor", 1.0)))

    # Noise Transformations
    if "gaussian_noise" in techniques:
        var = float(params.get("gaussian_variance", 0.01))
        image = add_gaussian_noise(image, var=var)
    if "salt_pepper_noise" in techniques:
        amount = float(params.get("sap_amount", 0.005)) # salt and pepper amount
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
        image = apply_random_erasing(image) # Uses default params in function

    # Mixup and Cutmix (requires another image from the uploaded set)
    other_image_for_mix = None
    if ("mixup" in techniques or "cutmix" in techniques) and files_in_dataset:
        possible_other_files = [f for f in files_in_dataset if f != current_image_filename]
        if not possible_other_files and len(files_in_dataset) > 0:
            possible_other_files = files_in_dataset # Use any file if current is the only one or not found

        if possible_other_files:
            other_filename = random.choice(possible_other_files)
            other_filepath = os.path.join(source_dataset_folder, other_filename)
            try:
                other_image_for_mix = Image.open(other_filepath)
            except Exception as e:
                print(f"Warning: Could not load other image {other_filepath} for mixup/cutmix: {e}")
                other_image_for_mix = None
        else:
            print("Warning: No other images available for mixup/cutmix.")

    if "mixup" in techniques and other_image_for_mix:
        alpha = float(params.get("mixup_alpha", 0.4))
        image = apply_mixup(image, other_image_for_mix, alpha)
    if "cutmix" in techniques and other_image_for_mix:
        image = apply_cutmix(image, other_image_for_mix)
    
    if other_image_for_mix: # Close the image if opened
        other_image_for_mix.close()

    return image

# --- End of Augmentation Functions ---

@app.route('/upload', methods=['POST'])
def upload_files():
    dataset = request.form.get('dataset')
    files = request.files.getlist('files')

    if not dataset:
        return "No dataset name provided", 400

    dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset))
    os.makedirs(dataset_folder, exist_ok=True)

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(dataset_folder, filename)
        file.save(filepath)

        if filename.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_folder)
                os.remove(filepath) # Remove zip after extraction
            except zipfile.BadZipFile:
                print(f"Bad zip file: {filepath}") # Keep the bad zip for inspection
                # Or return an error to the user:
                # return f"Uploaded file {filename} is not a valid ZIP file.", 400


    return jsonify({"message": "Files uploaded successfully", "dataset": dataset}), 200


@app.route('/datasets', methods=['GET'])
def list_datasets():
    datasets_info = []
    base_upload_folder = app.config['UPLOAD_FOLDER']
    for folder_name in os.listdir(base_upload_folder):
        folder_path = os.path.join(base_upload_folder, folder_name)
        if os.path.isdir(folder_path):
            files_in_folder = os.listdir(folder_path)
            image_files = [f for f in files_in_folder if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            metadata = augmentation_metadata.get(folder_name, {})
            datasets_info.append({
                'name': folder_name,
                'count': len(image_files),
                'files': files_in_folder, # List all files, not just images, for completeness
                'techniques': metadata.get("techniques", []),
                'parameters': metadata.get("parameters", {}),
                'augmented_zip': metadata.get("augmented_zip")
            })
    return jsonify(datasets_info)


@app.route('/augment', methods=['POST'])
def augment_dataset_route():
    data = request.get_json()
    dataset_name = secure_filename(data.get('datasetName'))
    techniques = data.get('techniques', [])
    parameters = data.get('parameters', {})

    if not dataset_name:
        return jsonify({'error': 'Dataset name not provided'}), 400
    if not techniques:
        return jsonify({'error': 'No augmentation techniques selected'}), 400


    source_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    target_folder_base = os.path.join(app.config['AUGMENTED_FOLDER'], dataset_name)
    
    # Create a unique subfolder for this augmentation run to avoid overwriting
    run_index = 0
    target_folder = os.path.join(target_folder_base, f"run_{run_index}")
    while os.path.exists(target_folder):
        run_index += 1
        target_folder = os.path.join(target_folder_base, f"run_{run_index}")
    os.makedirs(target_folder, exist_ok=True)


    if not os.path.isdir(source_folder):
        return jsonify({'error': f"Source dataset folder '{dataset_name}' not found."}), 404

    source_image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    augmented_file_names = []
    for filename in source_image_files:
        img_path = os.path.join(source_folder, filename)
        try:
            with Image.open(img_path) as img:
                # Pass necessary info for mixup/cutmix
                augmented_img = run_augmentations(
                    img.copy(), # Pass a copy to avoid issues with original image object
                    techniques,
                    parameters,
                    source_folder,          # For Mixup/Cutmix to load other images
                    source_image_files,     # List of image names in source_folder
                    filename                # Current image filename
                )
                
                # Ensure mode is suitable for saving (e.g., convert P mode if it has alpha)
                if augmented_img.mode == 'P' and 'transparency' in augmented_img.info:
                    augmented_img = augmented_img.convert("RGBA")
                elif augmented_img.mode == 'LA' or (augmented_img.mode == 'L' and 'transparency' in augmented_img.info): # Grayscale with alpha
                    augmented_img = augmented_img.convert("RGBA") # or handle alpha appropriately
                elif augmented_img.mode not in ['RGB', 'RGBA', 'L']: # L for grayscale
                     augmented_img = augmented_img.convert('RGB')


                # Determine save format based on original or default to PNG for augmentations
                base, ext = os.path.splitext(filename)
                save_filename = f"aug_{base}{ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.png'}"
                save_path = os.path.join(target_folder, save_filename)
                
                if ext.lower() in ['.jpg', '.jpeg']:
                    augmented_img.save(save_path, "JPEG", quality=95)
                else: # Default to PNG
                    augmented_img.save(save_path, "PNG")
                augmented_file_names.append(save_filename)

        except Exception as e:
            print(f"Error augmenting image {filename}: {e}")
            # Optionally, copy original if augmentation fails
            # import shutil
            # shutil.copy(img_path, os.path.join(target_folder, f"err_orig_{filename}"))


    # Zip the augmented folder (only the current run)
    zip_filename = f"{dataset_name}_augmented_run_{run_index}.zip"
    # Save zip inside the specific run's folder or one level up in AUGMENTED_FOLDER/dataset_name
    zip_path = os.path.join(target_folder_base, zip_filename) # Store zip one level up from run_x folder
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for aug_file in augmented_file_names: # os.listdir(target_folder)
            file_to_zip_path = os.path.join(target_folder, aug_file)
            if os.path.isfile(file_to_zip_path) and aug_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                 zipf.write(file_to_zip_path, aug_file)


    # Save metadata (associating with the base dataset name)
    # This could be extended to store a list of augmentation runs if needed
    augmentation_metadata[dataset_name] = {
        "techniques": techniques,
        "parameters": parameters,
        "augmented_zip": zip_filename, # This will point to the latest zip
        "last_augmented_run_folder": target_folder # Store the path to the actual image files
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(augmentation_metadata, f, indent=4)

    return jsonify({'message': 'Augmentation complete', 'zip_filename': zip_filename, 'augmented_files_path': target_folder })


@app.route('/uploads/<dataset>/<filename>')
def uploaded_file(dataset, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset)), secure_filename(filename))

@app.route('/augmented/<dataset>/<zipfilename>') # This serves the ZIP file
def serve_augmented_zip(dataset, zipfilename):
    # Zips are stored in AUGMENTED_FOLDER/dataset_name/zipfilename
    return send_from_directory(os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset)), secure_filename(zipfilename), as_attachment=True)

# To view individual augmented images if needed (requires knowing the run folder)
# This endpoint might need adjustment if you want to browse specific runs.
# For simplicity, the ZIP download is primary.
# If you want to show augmented images in UI before zipping, that's a different flow.
@app.route('/augmented_image/<dataset>/<run_folder>/<filename>')
def serve_augmented_image(dataset, run_folder, filename):
    # Path: augmented/dataset_name/run_folder/filename
    return send_from_directory(os.path.join(app.config['AUGMENTED_FOLDER'], secure_filename(dataset), secure_filename(run_folder)), secure_filename(filename))



@app.route('/api/download-csv/<filename>', methods=['GET'])
def download_csv(filename):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)