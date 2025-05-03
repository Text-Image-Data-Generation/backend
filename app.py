# import os
# import json
# import random
# import zipfile
# from io import BytesIO
# from PIL import Image
# from flask import Flask, request, render_template, send_file, redirect, url_for, session
# from werkzeug.utils import secure_filename

# from config import UPLOAD_FOLDER, AUGMENTED_FOLDER, SECRET_KEY
# from utils import allowed_file
# from augmentations.pipeline import apply_augmentations

# app = Flask(__name__)
# app.secret_key = SECRET_KEY
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# # Ensure the upload and augmented directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# def handle_zip_file(file, upload_folder):
#     # Save the uploaded zip file temporarily
#     zip_filename = secure_filename(file.filename)
#     zip_filepath = os.path.join(upload_folder, zip_filename)
#     file.save(zip_filepath)
    
#     # Unzip the file
#     extracted_files = []
#     with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
#         # Extract all files to the upload folder
#         zip_ref.extractall(upload_folder)
#         extracted_files = zip_ref.namelist()
    
#     # Remove the zip file after extracting
#     os.remove(zip_filepath)
    
#     # Filter out image files (jpg, png, jpeg, gif)
#     image_files = [f for f in extracted_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
#     return image_files

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist('images')
#     uploaded_images = []

#     for file in files:
#         if file and allowed_file(file.filename):
#             if file.filename.endswith('.zip'):
#                 # Handle the zip file and extract images
#                 uploaded_images.extend(handle_zip_file(file, app.config['UPLOAD_FOLDER']))
#             else:
#                 # Handle a regular image file
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)
#                 uploaded_images.append(filename)

#     session['uploaded_files'] = uploaded_images
#     return redirect(url_for('select_augmentations'))

# @app.route('/select_augmentations', methods=['GET', 'POST'])
# def select_augmentations():
#     if request.method == 'POST':
#         augmentations = request.form.to_dict()
#         session['augmentations'] = augmentations
#         uploaded_images = len(session.get('uploaded_files', []))
#         return render_template('set_augmentation_count.html',
#                                augmentations=augmentations,
#                                uploaded_images=uploaded_images)
#     return render_template('select_augmentations.html')

# @app.route('/set_augmentation_count', methods=['POST'])
# def set_augmentation_count():
#     images_to_augment = int(request.form.get('images_to_augment', 1))
#     session['images_to_augment'] = images_to_augment
#     return redirect(url_for('apply_augmentations_route'))

# @app.route('/apply_augmentations', methods=['GET', 'POST'])
# def apply_augmentations_route():
#     uploaded_files = session.get('uploaded_files', [])
#     augmentations = session.get('augmentations', {})
#     images_to_augment = session.get('images_to_augment', len(uploaded_files))
#     params = augmentations.copy()
    
#     # Prepare techniques list (only those checked 'yes')
#     techniques = [key for key, value in augmentations.items() if value == 'yes']

#     # Create a new version folder for augmented images
#     existing_versions = [int(d.split('_')[-1])
#                          for d in os.listdir(app.config['AUGMENTED_FOLDER'])
#                          if d.startswith('version_')]
#     version_number = max(existing_versions + [0]) + 1
#     version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], f"version_{version_number}")
#     os.makedirs(version_folder, exist_ok=True)

#     total_augmented_images = 0

#     # If requested images exceed available images, adjust count
#     if images_to_augment > len(uploaded_files):
#         images_to_augment = len(uploaded_files)

#     images_to_augment_list = random.sample(uploaded_files, images_to_augment)

#     # Apply augmentations for each selected image
#     for filename in images_to_augment_list:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         image = Image.open(filepath)
#         augmented_image = apply_augmentations(image.copy(), techniques, params, uploaded_files)
#         file_root, file_ext = os.path.splitext(filename)
#         augmented_filename = f"{file_root}_aug{file_ext}"
#         augmented_filepath = os.path.join(version_folder, augmented_filename)
#         augmented_image.save(augmented_filepath)
#         total_augmented_images += 1

#     # Save metadata about this augmentation version
#     metadata = {
#         "total_augmented_images": total_augmented_images,
#         "images_to_augment": images_to_augment,
#         "selected_augmentations": techniques,
#         "augmentation_params": params
#     }
#     with open(os.path.join(version_folder, "metadata.json"), 'w') as f:
#         json.dump(metadata, f, indent=4)

#     # Clear session data and remove uploaded images
#     session.pop('uploaded_files', None)
#     session.pop('augmentations', None)
#     session.pop('images_to_augment', None)
#     for file in os.listdir(app.config['UPLOAD_FOLDER']):
#         os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

#     return redirect(url_for('download'))

# @app.route('/download')
# def download():
#     versions = {}
#     for folder in os.listdir(app.config['AUGMENTED_FOLDER']):
#         if folder.startswith('version_'):
#             version_number = int(folder.split('_')[-1])
#             metadata_path = os.path.join(app.config['AUGMENTED_FOLDER'], folder, "metadata.json")
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'r') as f:
#                     metadata = json.load(f)
#             else:
#                 metadata = {"total_augmented_images": 0,
#                             "selected_augmentations": [],
#                             "augmentation_params": {}}
#             versions[version_number] = metadata
#     sorted_versions = dict(sorted(versions.items(), reverse=True))
#     return render_template('download.html', versions=sorted_versions)

# @app.route('/download_zip/<version>')
# def download_zip(version):
#     if not version.startswith("version_"):
#         version = f"version_{version}"
#     version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], version)
#     zip_filename = f"augmented_images_{version.split('_')[-1]}.zip"

#     memory_file = BytesIO()
#     with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(version_folder):
#             for file in files:
#                 if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                     file_path = os.path.join(root, file)
#                     zipf.write(file_path, os.path.relpath(file_path, version_folder))
#         metadata_path = os.path.join(version_folder, "metadata.json")
#         if os.path.exists(metadata_path):
#             zipf.write(metadata_path, os.path.relpath(metadata_path, version_folder))
#         else:
#             return f"Metadata file not found in {version_folder}", 404
#     memory_file.seek(0)
#     return send_file(memory_file, download_name=zip_filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)





import os
import json
import random
import zipfile
from io import BytesIO
from PIL import Image
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from ctgan import CTGANSynthesizer

from config import UPLOAD_FOLDER, AUGMENTED_FOLDER, SECRET_KEY
from utils import allowed_file
from augmentations.pipeline import apply_augmentations

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUGMENTED_FOLDER'] = AUGMENTED_FOLDER

# Ensure the upload and augmented directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# Function for handling zip files
def handle_zip_file(file, upload_folder):
    zip_filename = secure_filename(file.filename)
    zip_filepath = os.path.join(upload_folder, zip_filename)
    file.save(zip_filepath)
    
    extracted_files = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(upload_folder)
        extracted_files = zip_ref.namelist()
    
    os.remove(zip_filepath)
    
    image_files = [f for f in extracted_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    return image_files

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    uploaded_images = []

    for file in files:
        if file and allowed_file(file.filename):
            if file.filename.endswith('.zip'):
                uploaded_images.extend(handle_zip_file(file, app.config['UPLOAD_FOLDER']))
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_images.append(filename)

    session['uploaded_files'] = uploaded_images
    return redirect(url_for('select_augmentations'))

# Route to select augmentations
@app.route('/select_augmentations', methods=['GET', 'POST'])
def select_augmentations():
    if request.method == 'POST':
        augmentations = request.form.to_dict()
        session['augmentations'] = augmentations
        uploaded_images = len(session.get('uploaded_files', []))
        return render_template('set_augmentation_count.html',
                               augmentations=augmentations,
                               uploaded_images=uploaded_images)
    return render_template('select_augmentations.html')

# Route to set augmentation count
@app.route('/set_augmentation_count', methods=['POST'])
def set_augmentation_count():
    images_to_augment = int(request.form.get('images_to_augment', 1))
    session['images_to_augment'] = images_to_augment
    return redirect(url_for('apply_augmentations_route'))

# Route to apply augmentations
@app.route('/apply_augmentations', methods=['GET', 'POST'])
def apply_augmentations_route():
    uploaded_files = session.get('uploaded_files', [])
    augmentations = session.get('augmentations', {})
    images_to_augment = session.get('images_to_augment', len(uploaded_files))
    params = augmentations.copy()
    
    techniques = [key for key, value in augmentations.items() if value == 'yes']

    existing_versions = [int(d.split('_')[-1])
                         for d in os.listdir(app.config['AUGMENTED_FOLDER'])
                         if d.startswith('version_')]
    version_number = max(existing_versions + [0]) + 1
    version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], f"version_{version_number}")
    os.makedirs(version_folder, exist_ok=True)

    total_augmented_images = 0

    if images_to_augment > len(uploaded_files):
        images_to_augment = len(uploaded_files)

    images_to_augment_list = random.sample(uploaded_files, images_to_augment)

    for filename in images_to_augment_list:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = Image.open(filepath)
        augmented_image = apply_augmentations(image.copy(), techniques, params, uploaded_files)
        file_root, file_ext = os.path.splitext(filename)
        augmented_filename = f"{file_root}_aug{file_ext}"
        augmented_filepath = os.path.join(version_folder, augmented_filename)
        augmented_image.save(augmented_filepath)
        total_augmented_images += 1

    metadata = {
        "total_augmented_images": total_augmented_images,
        "images_to_augment": images_to_augment,
        "selected_augmentations": techniques,
        "augmentation_params": params
    }
    with open(os.path.join(version_folder, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    session.pop('uploaded_files', None)
    session.pop('augmentations', None)
    session.pop('images_to_augment', None)
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

    return redirect(url_for('download'))

# Route to download augmented images
@app.route('/download')
def download():
    versions = {}
    for folder in os.listdir(app.config['AUGMENTED_FOLDER']):
        if folder.startswith('version_'):
            version_number = int(folder.split('_')[-1])
            metadata_path = os.path.join(app.config['AUGMENTED_FOLDER'], folder, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"total_augmented_images": 0,
                            "selected_augmentations": [],
                            "augmentation_params": {}}
            versions[version_number] = metadata
    sorted_versions = dict(sorted(versions.items(), reverse=True))
    return render_template('download.html', versions=sorted_versions)

# Route to download a zip of augmented images
@app.route('/download_zip/<version>')
def download_zip(version):
    if not version.startswith("version_"):
        version = f"version_{version}"
    version_folder = os.path.join(app.config['AUGMENTED_FOLDER'], version)
    zip_filename = f"augmented_images_{version.split('_')[-1]}.zip"

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(version_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, version_folder))
        metadata_path = os.path.join(version_folder, "metadata.json")
        if os.path.exists(metadata_path):
            zipf.write(metadata_path, os.path.relpath(metadata_path, version_folder))
        else:
            return f"Metadata file not found in {version_folder}", 404
    memory_file.seek(0)
    return send_file(memory_file, download_name=zip_filename, as_attachment=True)

# Route to generate synthetic data using CTGAN
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_path = data['input_path']
    output_path = data['output_path']
    num_samples = int(data.get('n', 200))
    epochs = int(data.get('epochs', 5))

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        return {'error': 'Unsupported file format. Use .csv or .xlsx.'}, 400

    discrete_columns = df.columns.tolist()

    ctgan = CTGANSynthesizer(epochs=epochs)
    ctgan.fit(df, discrete_columns)
    samples = ctgan.sample(num_samples)

    if input_path.endswith('.csv'):
        output_file = os.path.join(output_path, 'generated_data.csv')
        samples.to_csv(output_file, index=False)
    else:
        output_file = os.path.join(output_path, 'generated_data.xlsx')
        samples.to_excel(output_file, index=False)

    return {'output_file': output_file}

if __name__ == '__main__':
    app.run(debug=True,port=5001)
