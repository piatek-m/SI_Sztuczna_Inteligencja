import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data.transforms import test_transform
from src.data.dataset import get_classes
from src.models.model import build_model

classes = get_classes()
num_classes = len(classes)
model = build_model(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

test_data = datasets.ImageFolder(DATA_DIR / "val", transform=test_transform)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
print(f"Test set: {len(test_data)} images across {len(classes)} classes")

all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images_gpu = images.to(DEVICE)
        outputs = model(images_gpu)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_images.extend(images)

print(f"Total predictions: {len(all_preds)}")

report = classification_report(all_labels, all_preds, target_names=classes)
print(report)
with open(ARTIFACTS_DIR / "evaluation_report.txt", "w") as f:
    f.write(str(report))

conf_mtrx = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(
    conf_mtrx,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png")
plt.show()


def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return img.clip(0, 1)


def plot_misclassified(true_class_idx):
    items = [
        (img, pred)
        for img, true, pred in zip(all_images, all_labels, all_preds)
        if true == true_class_idx and true != pred
    ]
    if not items:
        return  # skip classes with no errors

    cols = min(len(items), 5)
    rows = math.ceil(len(items) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (img, pred) in zip(axes, items):
        ax.imshow(unnormalize(img))
        ax.set_title(f"Predicted: {classes[pred]}", fontsize=12)
        ax.axis("off")

    for ax in axes[len(items) :]:
        ax.axis("off")

    fig.suptitle(f"True class: {classes[true_class_idx]}")
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "misclassified_images" / f"{classes[true_class_idx]}.png"
    )
    plt.show()


for idx in range(len(classes)):
    plot_misclassified(idx)
