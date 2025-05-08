# app.py
import os
import zipfile
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

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

# Augmentation functions
def rotate_image(image, angle=90):
    return image.rotate(angle, expand=True)

def scale_image(image, scale=1.5):
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)))

def flip_horizontal(image):
    return ImageOps.mirror(image)

def flip_vertical(image):
    return ImageOps.flip(image)

def apply_augmentations(image, augmentations):
    for aug in augmentations:
        if aug == 'rotate':
            image = rotate_image(image)
        elif aug == 'scale':
            image = scale_image(image)
        elif aug == 'flip_horizontal':
            image = flip_horizontal(image)
        elif aug == 'flip_vertical':
            image = flip_vertical(image)
    return image

@app.route('/upload', methods=['POST'])
def upload_files():
    dataset = request.form.get('dataset')
    files = request.files.getlist('files')

    if not dataset:
        return "No dataset name provided", 400

    dataset_folder = os.path.join(UPLOAD_FOLDER, secure_filename(dataset))
    os.makedirs(dataset_folder, exist_ok=True)

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(dataset_folder, filename)
        file.save(filepath)

        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
            os.remove(filepath)

    return "Files uploaded", 200

@app.route('/datasets', methods=['GET'])
def list_datasets():
    datasets = []
    for folder in os.listdir(UPLOAD_FOLDER):
        folder_path = os.path.join(UPLOAD_FOLDER, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            metadata = augmentation_metadata.get(folder, {})
            datasets.append({
                'name': folder,
                'count': len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]),
                'files': files,
                'augmentations': metadata.get("augmentations", []),
                'augmented_zip': metadata.get("augmented_zip")
            })
    return jsonify(datasets)

@app.route('/augment', methods=['POST'])
def augment_dataset():
    data = request.get_json()
    dataset_name = secure_filename(data.get('datasetName'))
    augmentations = data.get('augmentations', [])

    source_folder = os.path.join(UPLOAD_FOLDER, dataset_name)
    target_folder = os.path.join(AUGMENTED_FOLDER, dataset_name)
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                augmented_img = apply_augmentations(img, augmentations)
                save_path = os.path.join(target_folder, f"aug_{filename}")
                augmented_img.save(save_path)

    # Zip the augmented folder
    zip_filename = f"{dataset_name}_augmented.zip"
    zip_path = os.path.join(target_folder, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(target_folder):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                zipf.write(os.path.join(target_folder, file), file)

    # Save metadata
    augmentation_metadata[dataset_name] = {
        "augmentations": augmentations,
        "augmented_zip": zip_filename
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(augmentation_metadata, f)

    return jsonify({'message': 'Augmentation complete', 'zip': zip_filename})

@app.route('/uploads/<dataset>/<filename>')
def uploaded_file(dataset, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, secure_filename(dataset)), filename)

@app.route('/augmented/<dataset>/<filename>')
def augmented_file(dataset, filename):
    return send_from_directory(os.path.join(AUGMENTED_FOLDER, secure_filename(dataset)), filename)

from flask import send_from_directory

@app.route('/download/<dataset_name>/<filename>')
def download_file(dataset_name, filename):
    augmented_dir = os.path.join('augmented', dataset_name)
    return send_from_directory(augmented_dir, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
