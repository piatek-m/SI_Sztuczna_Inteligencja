import torch
import json
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parents[1]
MODELS_DIR = PROJECT_DIR / "models" / "checkpoints"
DATA_DIR = PROJECT_DIR / "Food_and_Vegetables"
ASSETS_DIR = PROJECT_DIR / "assets"

# Hiperparameters
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

# Transforms
IMAGENET_SIZE = (224, 244)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE = 256
CROP = 224

# CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
