# config.py
import os

# Flask App Configuration
# IMPORTANT: Keep this key secret and static in production!
# For development, this static key is fine.
SECRET_KEY = 'your-very-secret-and-static-key-for-flask-sessions' 
API_PORT = 5001

# Folder Configurations
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploaded_images_server')
AUGMENTED_FOLDER = os.path.join(BASE_DIR, 'augmented_images_server')

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# MongoDB Configuration (Placeholder - not used in current file-based implementation)
# MONGO_URI = "mongodb://localhost:5002/" # Assuming your MongoDB is on 5002
# MONGO_DATABASE = "image_augmentation_db"

DEBUG = True # Set to False in production
