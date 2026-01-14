"""
Configuration settings for the corner detection model.
"""
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_split_v2")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Model output directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_model", "saved_models")

# Feature engineering parameters
WINDOW_SIZE = 5  # Number of neighboring points for local features
SMOOTHING_WINDOW = 3  # Window size for coordinate smoothing

# Model parameters
RANDOM_STATE = 42
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
