import os
import zipfile
import json
import random
import torch
import sys
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

ENHANCED_FOLDER = 'enhanced'
ENHANCED_METADATA_FILE = 'enhancement_metadata.json'
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

if os.path.exists(ENHANCED_METADATA_FILE) and os.path.getsize(ENHANCED_METADATA_FILE) > 0:
    try:
        with open(ENHANCED_METADATA_FILE, 'r') as f:
            enhancement_metadata = json.load(f)
    except json.JSONDecodeError:
        print("Warning: enhancement_metadata.json is malformed. Using empty metadata.")
        enhancement_metadata = {}
else:
    enhancement_metadata = {}



# Ensure results folder exists
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(csv_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

if os.path.exists(METADATA_FILE) and os.path.getsize(METADATA_FILE) > 0:
    try:
        with open(METADATA_FILE, 'r') as f:
            augmentation_metadata = json.load(f)
    except json.JSONDecodeError:
        print("Warning: augmentation_metadata.json is malformed. Using empty metadata.")
        augmentation_metadata = {}
else:
    augmentation_metadata = {}



GRADIO_URL_IMAGES = os.getenv("GRADIO_URL_IMAGES")
IMAGES_DIR = "generated_images"
META_FILE = "image_creation_metadata.json"

os.makedirs(IMAGES_DIR, exist_ok=True)
if not os.path.exists(META_FILE):
    with open(META_FILE, "w") as f:
        json.dump([], f)



#hero ---

# Adjust this path based on your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), 'ESRGAN'))
import RRDBNet_arch as arch



CSV_META_PATH = os.path.join(RESULT_DIR, "csv_meta.json")

def load_csv_meta():
    if not os.path.exists(CSV_META_PATH):
        with open(CSV_META_PATH, 'w') as f:
            json.dump([], f)
    with open(CSV_META_PATH, 'r') as f:
        return json.load(f)

def save_csv_meta(meta):
    with open(CSV_META_PATH, 'w') as f:
        json.dump(meta, f, indent=4)



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
    print(file)
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
            handle_file(temp_input_path),
            epochs,
            num_samples,
            api_name="/predict"
        )

        if not isinstance(prediction_result_path, str) or not os.path.exists(prediction_result_path):
            print(f"Gradio client did not return a valid file path. Result: {prediction_result_path}")
            return jsonify({"error": "Synthetic data generation failed or returned unexpected result."}), 500
        
        base, ext = os.path.splitext(temp_input_filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        output_filename = f"{base}_gen_{timestamp}{ext}"
        output_path = os.path.join(RESULT_DIR, output_filename)
        shutil.copy(prediction_result_path, output_path)

        # Save metadata
        meta = load_csv_meta()
        meta.append({
            "filename": output_filename,
            "original_file": temp_input_filename,
            "samples": num_samples,
            "epochs": epochs,
            "timestamp": timestamp
        })
        save_csv_meta(meta)

        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"message": "Synthetic data generated successfully.", "output_file": output_filename})
    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/get-csv-history', methods=['GET'])
def get_csv_history():
    meta = load_csv_meta()
    return jsonify(meta)


@app.route('/download-csv/<filename>', methods=['GET'])
def download_file(filename):
    safe_filename = secure_filename(filename)
    if not safe_filename == filename:
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(RESULT_DIR, safe_filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found or access denied"}), 404
    return send_file(filepath, as_attachment=True)


def enhance_esrgan_image(input_path, output_path, model_path, device='cpu'):
    device = torch.device(device)
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    print(f'Loaded model from {model_path}')
    print(f'Enhancing image: {input_path}')

    # Read and preprocess image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_path}")
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # Save result
    cv2.imwrite(output_path, output)
    print(f'Saved enhanced image to: {output_path}')


@app.route('/upload_and_enhance', methods=['POST'])
def upload_and_enhance_images():
    dataset = request.form.get('dataset')
    files = request.files.getlist('files')
    if not dataset:
        return jsonify({"error": "Dataset name required"}), 400
    dataset_safe = secure_filename(dataset)
    base_path = os.path.join(ENHANCED_FOLDER, dataset_safe)
    originals_path = os.path.join(base_path, 'originals')
    predictions_path = os.path.join(base_path, 'predictions')
    zip_path = os.path.join(base_path, f"{dataset_safe}_enhanced.zip")
    os.makedirs(originals_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)

    model_path = os.path.join("ESRGAN", "models", "RRDB_ESRGAN_x4.pth")

    enhanced_files = []
    for file in files:
        filename = secure_filename(file.filename)
        input_path = os.path.join(originals_path, filename)
        file.save(input_path)

        output_filename = f"{os.path.splitext(filename)[0]}_ESRGAN.png"
        output_path = os.path.join(predictions_path, output_filename)
        try:
            enhance_esrgan_image(input_path, output_path, model_path)
            enhanced_files.append(output_filename)
        except Exception as e:
            print(f"Enhancement failed for {filename}: {e}")

    # Save ZIP of enhanced images
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in enhanced_files:
            zipf.write(os.path.join(predictions_path, f), arcname=f)

    enhancement_metadata[dataset_safe] = {
        "dataset": dataset_safe,
        "original_count": len(files),
        "enhanced_count": len(enhanced_files),
        "zip": f"{dataset_safe}_enhanced.zip"
    }
    with open(ENHANCED_METADATA_FILE, 'w') as f:
        json.dump(enhancement_metadata, f, indent=4)

    return jsonify({
        "message": "Upload and enhancement complete",
        "dataset": dataset_safe,
        "enhanced_count": len(enhanced_files),
        "zip_filename": f"{dataset_safe}_enhanced.zip"
    })



@app.route('/enhanced_datasets', methods=['GET'])
def list_enhanced_datasets():
    return jsonify(list(enhancement_metadata.values()))

@app.route('/enhanced_zip/<dataset>')
def serve_enhanced_zip(dataset):
    safe = secure_filename(dataset)
    zip_path = os.path.join(ENHANCED_FOLDER, safe, f"{safe}_enhanced.zip")
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    return jsonify({"error": "ZIP not found"}), 404




@app.route('/generate-images', methods=['POST'])
def generate_images():
    truncation = float(request.form.get('truncation', 0.7))
    seed_start = int(request.form.get('seed_start', 0))
    seed_end = int(request.form.get('seed_end', 24))

    try:
        client = Client(GRADIO_URL_IMAGES)
        zip_path = client.predict(truncation, seed_start, seed_end, api_name="/predict")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"images_{seed_start}_{seed_end}_{timestamp}.zip"
        saved_path = os.path.join(IMAGES_DIR, filename)
        shutil.copy(zip_path, saved_path)

        # Save metadata
        with open(META_FILE, "r+") as meta_file:
            data = json.load(meta_file)
            data.append({
                "timestamp": timestamp,
                "filename": filename,
                "seed_start": seed_start,
                "seed_end": seed_end,
                "truncation": truncation
            })
            meta_file.seek(0)
            json.dump(data, meta_file, indent=2)

        return jsonify({"message": "Images generated", "filename": filename})

    except Exception as e:
        print("Image generation failed:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/image-generation-history', methods=['GET'])
def get_image_generation_history():
    if not os.path.exists(META_FILE):
        return jsonify([])
    with open(META_FILE, "r") as meta_file:
        data = json.load(meta_file)
    return jsonify(data)


@app.route('/download-image-zip/<filename>', methods=['GET'])
def download_image_zip(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)