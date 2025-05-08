# utils.py
from config import ALLOWED_EXTENSIONS

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    Args:
        filename (str): The name of the file.
    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
