import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 36

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)
