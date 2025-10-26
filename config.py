import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'forest_watch'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'AladinoBuhari')
}

# Data parameters
CANOPY_THRESHOLD = 30  # Default: >30% canopy cover
YEARS_RANGE = (2001, 2024)

# Analysis parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# API configuration (for future use)
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))