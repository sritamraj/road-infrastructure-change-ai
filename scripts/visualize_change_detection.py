from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.change_detection.dataset import LEVIRChangeDataset
from src.change_detection.models import SiameseChangeNet


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = (
        "outputs/checkpoints/"
        "siamese_change_weighted.pt"
    )

    threshold = 0.45

    output_dir = Path(
        "outputs/change_visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = LEVIRChangeDataset(
        split="test",
        size=256,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = SiameseChangeNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print(f"Device: {device}")
    print(f"Test samples: {len(dataset)}")
    print(f"Threshold: {threshold}")

    selected_indices = [0, 1, 2, 3, 4]

    for index, (image_a, image_b, target) in enumerate(loader):

        if index not in selected_indices:
            continue

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        target = target.to(device)

        with torch.no_grad():

            logits = model(
                image_a,
                image_b,
            )

            probability = torch.sigmoid(
                logits
            )

        prediction = (
            probability >= threshold
        )

        image_a_np = (
            image_a[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        image_b_np = (
            image_b[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        mean = np.array(
            [0.485, 0.456, 0.406]
        )

        std = np.array(
            [0.229, 0.224, 0.225]
        )

        image_a_np = (
            image_a_np * std + mean
        )

        image_b_np = (
            image_b_np * std + mean
        )

        image_a_np = np.clip(
            image_a_np,
            0,
            1,
        )

        image_b_np = np.clip(
            image_b_np,
            0,
            1,
        )

        target_np = (
            target[0, 0]
            .cpu()
            .numpy()
        )

        probability_np = (
            probability[0, 0]
            .cpu()
            .numpy()
        )

        prediction_np = (
            prediction[0, 0]
            .cpu()
            .numpy()
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
        )

        axes[0].imshow(image_a_np)
        axes[0].set_title("T1 Image")
        axes[0].axis("off")

        axes[1].imshow(image_b_np)
        axes[1].set_title("T2 Image")
        axes[1].axis("off")

        axes[2].imshow(target_np)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        axes[3].imshow(prediction_np)
        axes[3].set_title(
            f"Prediction (threshold={threshold})"
        )
        axes[3].axis("off")

        fig.suptitle(
            f"LEVIR-CD Test Sample {index}",
            fontsize=14,
        )

        fig.tight_layout()

        output_path = (
            output_dir
            / f"change_sample_{index}.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )

        probability_path = (
            output_dir
            / f"change_sample_{index}_probability.png"
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            probability_np,
            vmin=0,
            vmax=1,
        )

        plt.colorbar(
            label="Change probability"
        )

        plt.title(
            f"Change Probability - Sample {index}"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            probability_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved: {probability_path}"
        )


if __name__ == "__main__":
    main()