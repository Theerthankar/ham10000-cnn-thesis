"""Train and eval transforms.

Image size is a parameter because EfficientNet-B3 is run twice, once at 224 to
match the other two backbones and once at its native 300. Everything else in
the pipeline is identical between those two runs, so the pair isolates the
effect of input resolution on its own.

The augmentation list itself never changes across experiments. It is part of
the fixed protocol, not a variable.
"""

from torchvision import transforms

from .constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def train_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
