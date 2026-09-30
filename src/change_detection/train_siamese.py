import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.change_detection.dataset import LEVIRChangeDataset
from src.change_detection.models import SiameseChangeNet
from src.data.common import load_yaml


def weighted_dice_bce_loss(
    logits,
    target,
    positive_weight=4.0,
):
    # Give changed pixels more importance because
    # they are much less common than unchanged pixels.
    pos_weight = torch.tensor(
        [positive_weight],
        device=logits.device,
    )

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + 1e-6)
        / (denominator + 1e-6)
    ).mean()

    return bce + (1.0 - dice)


def calculate_metrics(
    logits,
    target,
    threshold=0.15,
):
    prediction = (
        torch.sigmoid(logits) >= threshold
    )

    target_bool = target >= 0.5

    tp = (
        prediction & target_bool
    ).sum().item()

    tn = (
        ~prediction & ~target_bool
    ).sum().item()

    fp = (
        prediction & ~target_bool
    ).sum().item()

    fn = (
        ~prediction & target_bool
    ).sum().item()

    precision = (
        tp / (tp + fp + 1e-8)
    )

    recall = (
        tp / (tp + fn + 1e-8)
    )

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + 1e-8)
    )

    iou = (
        tp
        / (tp + fp + fn + 1e-8)
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    threshold=0.15,
    positive_weight=4.0,
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for image_a, image_b, target in loader:

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits = model(
                image_a,
                image_b,
            )

            loss = weighted_dice_bce_loss(
                logits,
                target,
                positive_weight=positive_weight,
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item()
            * image_a.size(0)
        )

        metrics = calculate_metrics(
            logits.detach(),
            target,
            threshold=threshold,
        )

        total_tp += metrics["tp"]
        total_tn += metrics["tn"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

    precision = (
        total_tp
        / (total_tp + total_fp + 1e-8)
    )

    recall = (
        total_tp
        / (total_tp + total_fn + 1e-8)
    )

    dice = (
        2.0 * total_tp
        / (
            2.0 * total_tp
            + total_fp
            + total_fn
            + 1e-8
        )
    )

    iou = (
        total_tp
        / (
            total_tp
            + total_fp
            + total_fn
            + 1e-8
        )
    )

    return {
        "loss": (
            total_loss
            / len(loader.dataset)
        ),
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Siamese change detection model.")
    parser.parse_args()

    config = load_yaml("configs/change_detection.yaml")
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["outputs"]

    torch.manual_seed(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    train_dataset = LEVIRChangeDataset(
        split="train",
        size=data_config["image_size"],
        train=True,
        root=data_config["root"],
    )

    val_dataset = LEVIRChangeDataset(
        split="val",
        size=data_config["image_size"],
        train=False,
        root=data_config["root"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
    )

    model = SiameseChangeNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    epochs = training_config["epochs"]

    evaluation_threshold = training_config["threshold"]

    output_dir = Path(
        output_config["checkpoint_dir"]
    )

    metrics_dir = Path(
        output_config["metrics_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_iou = -1.0

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Evaluation threshold: "
        f"{evaluation_threshold}"
    )

    print(
        "Positive-class weight:", training_config["positive_weight"]
    )

    print(
        "Epochs:",
        epochs,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
            threshold=evaluation_threshold,
            positive_weight=training_config["positive_weight"],
        )

        with torch.no_grad():

            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                training=False,
                threshold=evaluation_threshold,
                positive_weight=training_config["positive_weight"],
            )

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "threshold": evaluation_threshold,
            "positive_weight": training_config["positive_weight"],
            "epoch_seconds": epoch_time,
        }

        history.append(record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"time={epoch_time:.1f}s | "
            f"train_loss="
            f"{train_metrics['loss']:.4f} | "
            f"val_loss="
            f"{val_metrics['loss']:.4f} | "
            f"val_precision="
            f"{val_metrics['precision']:.4f} | "
            f"val_recall="
            f"{val_metrics['recall']:.4f} | "
            f"val_dice="
            f"{val_metrics['dice']:.4f} | "
            f"val_iou="
            f"{val_metrics['iou']:.4f}"
        )

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "threshold": evaluation_threshold,
                    "positive_weight": training_config["positive_weight"],
                },
                output_dir
                / "siamese_change_weighted.pt",
            )

            print(
                "  New best checkpoint saved."
            )

    history_path = (
        metrics_dir
        / "change_detection_weighted_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print(
        f"Best validation IoU: "
        f"{best_iou:.4f}"
    )

    print(
        "Checkpoint saved:",
        output_dir
        / "siamese_change_weighted.pt",
    )

    print(
        "History saved:",
        history_path,
    )


if __name__ == "__main__":
    main()