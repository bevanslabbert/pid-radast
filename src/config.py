# src/config.py

import os

# Base project directory (useful for path resolution)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "data.pkl")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Classification model config
CLASSIFICATION_MODEL = {
    "input_dim": 128,
    "hidden_dim": 64,
    "output_dim": 5,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50
}

# Diffusion model config
DIFFUSION_MODEL = {
    "grid_size": 100,
    "time_steps": 1000,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "physics_loss_weight": 10.0
}

# Random seed
SEED = 42