import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import RoadDataset
    from .models import build_model
    from .losses import DiceBCELoss
    from ..data.common import load_yaml, seed_everything
    from ..evaluation.metrics import binary_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.segmentation.dataset import RoadDataset
    from src.segmentation.models import build_model
    from src.segmentation.losses import DiceBCELoss
    from src.data.common import load_yaml, seed_everything
    from src.evaluation.metrics import binary_metrics


def run(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(images)
        logits = getattr(logits, "logits", logits)

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        all_targets.append(masks.detach().cpu())
        all_predictions.append(
            torch.sigmoid(logits).detach().cpu()
        )

    avg_loss = total_loss / len(loader.dataset)

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    metrics = binary_metrics(targets, predictions)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a U-Net road segmentation model."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to segmentation YAML configuration.",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    split_dir = Path(
        config["data"].get(
            "split_dir",
            "data/splits/deepglobe",
        )
    )

    train_split = Path(
        config["data"].get(
            "train_split",
            "train.txt",
        )
    )

    val_split = Path(
        config["data"].get(
            "val_split",
            "val.txt",
        )
    )

    train_dataset = RoadDataset(
        split_dir / train_split,
        size=config["data"]["image_size"],
        train=True,
    )

    val_dataset = RoadDataset(
        split_dir / val_split,
        size=config["data"]["image_size"],
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training split: {train_split}")
    print(f"Validation split: {val_split}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")

    model = build_model(
        config["training"]["model"]
    ).to(device)

    criterion = DiceBCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_dir = Path(
        config["outputs"]["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir = Path(
        config["outputs"]["metrics_dir"]
    )
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_iou = -1.0
    history = []

    for epoch in range(
        1,
        config["training"]["epochs"] + 1,
    ):
        print(
            f"\nEpoch {epoch}/"
            f"{config['training']['epochs']}"
        )

        train_loss, train_metrics = run(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_metrics = run(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        row.update(
            {
                f"train_{key}": value
                for key, value in train_metrics.items()
            }
        )

        row.update(
            {
                f"val_{key}": value
                for key, value in val_metrics.items()
            }
        )

        history.append(row)

        print(row)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]

            checkpoint_path = (
                checkpoint_dir / "unet_best.pt"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "metrics": val_metrics,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print(
                f"Saved best model. "
                f"Validation IoU: {best_iou:.4f}"
            )

    history_path = (
        metrics_dir / "segmentation_history.json"
    )

    history_path.write_text(
        json.dumps(history, indent=2)
    )

    print("\nTraining complete.")
    print(
        f"Best validation IoU: {best_iou:.4f}"
    )
    print(
        f"Checkpoint: {checkpoint_dir / 'unet_best.pt'}"
    )
    print(
        f"History: {history_path}"
    )


if __name__ == "__main__":
    main()