import time
import json
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import v2

from src.config import (
    ASSETS_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    FINETUNE_EPOCHS,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_LR,
    DEVICE,
    MIXUP_ALPHA,
)
from src.data.dataset import get_train_val_loaders, get_dataset
from src.models.model import build_model

run_start_time = time.time()

print(f"Training on: {DEVICE}")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda device")
print(f"HEAD_EPOCHS={CLASSIFIER_EPOCHS}")
print(f"FINETUNE_EPOCHS={FINETUNE_EPOCHS}")
print(f"STARTING_LR={CLASSIFIER_LR:.2e}")

# Data loading
train_subset, val_subset, classes = get_dataset()
train_loader, val_loader = get_train_val_loaders(train_subset, val_subset)

num_classes = len(classes)
print(f"Classes: {num_classes} | Train: {len(train_subset)} | Val: {len(val_subset)}")
with open(ASSETS_DIR / "class_names.json", "w") as f:
    json.dump(classes, f)

# Model
model = build_model(num_classes).to(DEVICE)


# --- Training logic ---

""" 
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0
    correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
    return avg_loss, accuracy """

mixup = v2.MixUp(num_classes=num_classes, alpha=MIXUP_ALPHA)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0
    correct = total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                images, labels = mixup(images, labels)
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)

            if training:
                targets = labels.argmax(dim=1)
            else:
                targets = labels

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    backbone_phase: bool,
):
    num_epochs = CLASSIFIER_EPOCHS
    checkpoint_suffix = ""
    print_epoch_prefix = ""
    if backbone_phase:
        num_epochs = FINETUNE_EPOCHS
        checkpoint_suffix = "_finetune"
        print_epoch_prefix = "[Finetune] "

    best_accuracy = 0
    smallest_loss = float("inf")
    for epoch in range(num_epochs):
        t0 = time.time()
        avg_train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        avg_val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(avg_val_loss)

        if avg_val_loss < smallest_loss:
            smallest_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                CHECKPOINTS_DIR / f"smallest_loss_model{checkpoint_suffix}.pth",
            )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(
                model.state_dict(),
                CHECKPOINTS_DIR / f"best_accuracy_model{checkpoint_suffix}.pth",
            )

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        last_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{print_epoch_prefix}Epoch {epoch+1}/{num_epochs} | Time: {time.time()-t0:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {last_lr:.2e}"
        )


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# History for plotting
history = {
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": [],
    "phase1_epochs": int,
}
history["phase1_epochs"] = CLASSIFIER_EPOCHS  # Divider

# Phase 1
print(
    f"Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
)
print(
    f"Training params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# Training optimization
classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=CLASSIFIER_LR)
classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    classifier_optimizer, mode="min", factor=0.5, patience=3
)

run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    classifier_optimizer,
    classifier_scheduler,
    DEVICE,
    backbone_phase=False,
)

# Phase 2
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "smallest_loss_model.pth"))

for param in model.parameters():
    param.requires_grad = True

print(
    f"Training params after unfreezing: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

finetine_lr = classifier_optimizer.param_groups[0]["lr"] * 1e-1
finetune_optimizer = torch.optim.Adam(
    [
        {"params": model.features.parameters(), "lr": finetine_lr},
        {"params": model.classifier.parameters(), "lr": finetine_lr * 1e1},
    ]
)
finetune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    finetune_optimizer, mode="min", factor=0.5, patience=3
)

run_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    finetune_optimizer,
    finetune_scheduler,
    DEVICE,
    backbone_phase=True,
)

shutil.copy(
    CHECKPOINTS_DIR / "best_accuracy_model_finetune.pth", MODELS_DIR / "best_model.pth"
)

run_duration = time.time() - run_start_time
# waow 🐱 such a cool function
duration_minutes, duration_seconds = divmod(run_duration, 60)
print(f"Full run time:{duration_minutes}m:{duration_seconds:.2f}s")

# Plotting
sum_epochs = CLASSIFIER_EPOCHS + FINETUNE_EPOCHS


def plot_history(history, plotting_loss: bool):
    epochs = range(1, sum_epochs + 1)

    train_values = "train_acc"
    val_values = "val_acc"
    plot_label = "Accuracy"
    y_scale = "linear"
    save_file_suffix = "accuracy"
    if plotting_loss:
        train_values = "train_loss"
        val_values = "val_loss"
        plot_label = "Loss"
        y_scale = "log"
        save_file_suffix = "loss"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[train_values], label="Train " + plot_label)
    plt.plot(epochs, history[val_values], label="Val " + plot_label)
    plt.axvline(
        x=history["phase1_epochs"], color="gray", linestyle="--", label="Finetune start"
    )
    plt.yscale(y_scale)
    plt.xlabel("Epoch")
    plt.ylabel(plot_label)
    plt.title("Training and validation " + plot_label)
    plt.legend()
    plt.savefig(ARTIFACTS_DIR / f"train_{save_file_suffix}.png", dpi=120)
    plt.show()


plot_history(history, plotting_loss=True)
plot_history(history, plotting_loss=False)
