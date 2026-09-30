from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRChangeDataset(Dataset):
    def __init__(self, split="train", size=256, train=False, root="data/raw/levir_cd"):
        self.root = Path(root) / split

        self.a_dir = self.root / "A"
        self.b_dir = self.root / "B"
        self.label_dir = self.root / "label"

        self.ids = sorted(
            p.stem
            for p in self.a_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        transforms = [
            A.Resize(
                height=size,
                width=size,
                interpolation=1,
                mask_interpolation=0,
            )
        ]

        if train:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )

        self.transform = A.Compose(
            transforms,
            additional_targets={"image_b": "image"},
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]

        a_path = next(self.a_dir.glob(sample_id + ".*"))
        b_path = next(self.b_dir.glob(sample_id + ".*"))
        label_path = next(self.label_dir.glob(sample_id + ".*"))

        image_a = np.array(
            Image.open(a_path).convert("RGB")
        )

        image_b = np.array(
            Image.open(b_path).convert("RGB")
        )

        label = np.array(
            Image.open(label_path).convert("L")
        )

        label = (label > 0).astype("float32")

        transformed = self.transform(
            image=image_a,
            image_b=image_b,
            mask=label,
        )

        image_a = transformed["image"]
        image_b = transformed["image_b"]
        label = transformed["mask"]

        image_a = image_a.astype(np.float32) / 255.0
        image_b = image_b.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image_a = (image_a - mean) / std
        image_b = (image_b - mean) / std

        image_a = torch.from_numpy(
            image_a.transpose(2, 0, 1)
        ).float()

        image_b = torch.from_numpy(
            image_b.transpose(2, 0, 1)
        ).float()

        label = torch.from_numpy(
            label[None]
        ).float()

        return image_a, image_b, label