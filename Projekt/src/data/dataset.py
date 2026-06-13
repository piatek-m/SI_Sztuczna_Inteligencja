import json
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR, ASSETS_DIR
from src.data.transforms import train_transform, test_transform
from src.config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


def get_classes():
    with open(ASSETS_DIR / "class_names.json") as f:
        return json.load(f)


def get_dataset():
    full_train_augmented = datasets.ImageFolder(
        DATA_DIR / "train", transform=train_transform
    )
    full_train_plain = datasets.ImageFolder(
        DATA_DIR / "train", transform=test_transform
    )

    targets = full_train_augmented.targets
    train_idx, val_idx = train_test_split(
        range(len(full_train_augmented)),
        test_size=0.15,
        stratify=targets,
        random_state=42,
    )

    train_data = Subset(full_train_augmented, train_idx)
    val_data = Subset(full_train_plain, val_idx)
    classes = full_train_plain.classes

    return train_data, val_data, classes


def get_train_val_loaders(train_data, val_data):
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader
