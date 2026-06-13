import torch
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parents[1]
MODELS_DIR = PROJECT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
DATA_DIR = PROJECT_DIR / "data"
ASSETS_DIR = PROJECT_DIR / "assets"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Hiperparameters
BATCH_SIZE = 16
CLASSIFIER_EPOCHS = 20
FINETUNE_EPOCHS = 10
CLASSIFIER_LR = 3e-4

# Transforms
IMAGENET_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE = 256
CROP = 224

# ROCm
NUM_WORKERS = 8
PIN_MEMORY = False  # only applies too CUDA ;c
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference
INFERENCE_INTERVAL = 1.5
