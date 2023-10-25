from pathlib import Path
import random
import shutil

import torch
import numpy as np
from sklearn.model_selection import train_test_split


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def copy_files(img_paths: Path, dest_path: Path):
    for img_path in img_paths:
        img_name = img_path.name
        shutil.copy(img_path, dest_path / img_name)


def split_celeba_dataset(dataset_path: Path):
    parent_path = dataset_path.parent

    (parent_path / "train").mkdir(exist_ok=True)
    (parent_path / "val").mkdir(exist_ok=True)
    (parent_path / "test").mkdir(exist_ok=True)

    img_paths = list(dataset_path.iterdir())
    n_imgs = len(img_paths)
    print(f"Number of celeba images: {len(img_paths)}")

    n_train_imgs = int(n_imgs * 0.8)
    n_val_imgs = int(n_imgs * 0.1)
    n_test_imgs = int(n_imgs * 0.1)

    train_images = img_paths[:n_train_imgs]
    val_images = img_paths[n_train_imgs:n_train_imgs+n_val_imgs]
    test_images = img_paths[n_train_imgs+n_val_imgs:]

    print(f"Number of train images: {len(train_images)}")
    print(f"Number of val images: {len(val_images)}")
    print(f"Number of test images: {len(test_images)}")

    copy_files(train_images, parent_path / "train")
    copy_files(val_images, parent_path / "val")
    copy_files(test_images, parent_path / "test")