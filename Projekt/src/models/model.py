import torch.nn as nn
from torchvision import models


def build_model(num_classes: int):
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )

    return model


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model
