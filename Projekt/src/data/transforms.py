from torchvision import transforms
from src.config import IMAGENET_MEAN, IMAGENET_STD, RESIZE, CROP

train_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE),
        transforms.RandomResizedCrop(CROP),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
