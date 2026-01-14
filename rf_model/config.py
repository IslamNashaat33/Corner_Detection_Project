"""
Configuration settings for the Random Forest corner detection model.
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
MODEL_DIR = os.path.join(PROJECT_ROOT, "rf_model", "saved_models")

# Feature engineering parameters
WINDOW_SIZE = 5  # Number of neighboring points for local features

# Random Forest parameters
RANDOM_STATE = 42
N_ESTIMATORS = 100  # Number of trees in the forest
MAX_DEPTH = 20  # Maximum depth of each tree
MIN_SAMPLES_SPLIT = 5  # Minimum samples required to split a node
MIN_SAMPLES_LEAF = 2  # Minimum samples required in a leaf node
N_JOBS = -1  # Use all available CPU cores for parallel training

# Feature selection parameters
FEATURE_SELECTION_ENABLED = True  # Whether to perform feature selection
FEATURE_SELECTION_THRESHOLD = "mean"  # Threshold: "mean", "median", or float
FEATURE_SELECTION_MAX_FEATURES = None  # Max features to keep (None = no limit)
