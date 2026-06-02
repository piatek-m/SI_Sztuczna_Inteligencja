import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Config
DATA_DIR = "Projekt/Food_and_Vegetables"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Trening na: {DEVICE}")

# Transforms
IMAGENET_SIZE = (224, 244)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGEMEAN_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize(IMAGENET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGEMEAN_STD),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(IMAGENET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGEMEAN_STD),
    ]
)

# Data loading
train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.ImageFolder(DATA_DIR + "/val", transform=test_transform)
test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_data.classes)
print(f"Number of classes: {num_classes}")
print(f"Sample classes: {train_data.classes[:5]}")
print(f"Training images: {len(train_data)}")
print(f"Test images: {len(test_data)}")


# Model definition
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(960, 1280),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(960, num_classes),
)
model = model.to(DEVICE)

print(
    f"Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
)
print(
    f"Training params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

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

    print(f"Epoch{epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# line 85, in <module>
#     outputs = model(images)
# venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
