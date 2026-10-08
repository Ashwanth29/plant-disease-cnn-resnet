import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 128
BATCH_SIZE = 32


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_datasets():

    train_dataset = datasets.ImageFolder(
        "dataset/train",
        transform=transform
    )

    validation_dataset = datasets.ImageFolder(
        "dataset/validation",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        "dataset/test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    )


if __name__ == "__main__":

    datasets_info = load_datasets()

    train_dataset = datasets_info[0]

    print("Number of training images:",
          len(train_dataset))

    print("Classes:",
          train_dataset.classes)