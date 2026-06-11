import json
import torch
from models.model import build_model
from src.config import PROJECT_DIR, ASSETS_DIR, MODELS_DIR, DEVICE

# Classes
with open(ASSETS_DIR / "class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)
num_classes = len(CLASS_NAMES)

model = build_model(num_classes)
model.load_state_dict(torch.load(MODELS_DIR / "fruits_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
