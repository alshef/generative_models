from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset


class CelebaDataset(Dataset):
    def __init__(self, path: Path, transform) -> None:
        self.img_paths = list(path.iterdir())
        self.transforms = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> Any:
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        image = self.transforms(image)
        return image


def get_dataloaders_celeba(dataroot: Path,
                           batch_size: int,
                           num_workers: int = 0,
                           train_transforms=None,
                           test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = CelebaDataset(dataroot / "train", transform=train_transforms)
    val_dataset = CelebaDataset(dataroot / "val", transform=test_transforms)
    test_dataset = CelebaDataset(dataroot / "test", transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
