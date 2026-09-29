import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    def __init__(self, split_file, size=512, train=False):
        self.root = Path("data/raw/deepglobe/train")

        ids = Path(split_file).read_text().splitlines()
        self.ids = [x.strip() for x in ids if x.strip()]

        if train:
            self.tf = A.Compose([
                A.Resize(
                    height=size,
                    width=size,
                    interpolation=1,
                    mask_interpolation=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(
                    height=size,
                    width=size,
                    interpolation=1,
                    mask_interpolation=0,
                ),
                A.Normalize(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sample_id = self.ids[i]

        image_path = (
            self.root / "images" / f"{sample_id}_sat.jpg"
        )

        mask_path = (
            self.root / "masks" / f"{sample_id}_mask.png"
        )

        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        mask_rgb = np.array(
            Image.open(mask_path).convert("RGB")
        )

        # White = road, black = background
        mask = np.all(mask_rgb == 255, axis=2).astype("float32")

        transformed = self.tf(
            image=image,
            mask=mask,
        )

        image = transformed["image"]
        mask = transformed["mask"]

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = torch.from_numpy(
            mask[None]
        ).float()

        return image, mask