from torchvision import transforms
from src.config import IMAGENET_MEAN, IMAGENET_STD, RESIZE, CROP

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(CROP),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
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
