# Configuration variables

UPLOAD_TEMP_FOLDER = 'uploaded_images_temp' # New temporary folder for uploads
AUGMENTED_FOLDER = 'augmented_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
SECRET_KEY = 'your_very_secret_and_random_key_here' # CHANGE THIS IN PRODUCTION
# Add frontend origin for CORS
FRONTEND_URL = 'http://localhost:3000' # Default Create React App URL