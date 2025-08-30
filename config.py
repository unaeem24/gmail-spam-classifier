import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model files
MODEL_FILE = os.path.join(MODEL_DIR, 'email_classifier.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'feature_vectorizer.pkl')

# GAPI configuration
GMAIL_CREDENTIALS_FILE = os.path.join(BASE_DIR, 'credentials.json')
GMAIL_TOKEN_FILE = os.path.join(BASE_DIR, 'token.json')

# Categories
CATEGORIES = ['work', 'spam', 'promotions', 'social']
CATEGORY_COLORS = {
    'work': 'blue',
    'spam': 'red',
    'promotions': 'orange',
    'social': 'green'
}

# Feature extraction
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5