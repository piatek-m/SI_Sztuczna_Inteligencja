import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image
import json

from src.config import DATA_DIR, ASSETS_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, LR
from data.transforms import train_transform, test_transform
from models.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")


# Data loading
def rgb_loader(path):
    with Image.open(path) as img:
        return img.convert("RGBA").convert("RGB")


train_data = datasets.ImageFolder(
    DATA_DIR + "train",
    transform=train_transform,
    loader=rgb_loader,
)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.ImageFolder(
    DATA_DIR + "val",
    transform=test_transform,
    loader=rgb_loader,
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


num_classes = len(train_data.classes)
print(f"Classes: {num_classes} | Train: {len(train_data)} | Test: {len(test_data)}")

with open(ASSETS_DIR + "class_names.json", "w") as f:
    json.dump(train_data.classes, f)

# Model
model = build_model(num_classes).to(DEVICE)

print(
    f"Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
)
print(
    f"Training params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
    )

print("Model weights:")
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")

torch.save(model.state_dict(), MODELS_DIR / "checkpoints/fruits_model.pth")
