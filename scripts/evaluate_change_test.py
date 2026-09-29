import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.change_detection.dataset import LEVIRChangeDataset
from src.change_detection.models import SiameseChangeNet


def calculate_metrics(logits, target, threshold):

    prediction = torch.sigmoid(logits) >= threshold
    target_bool = target >= 0.5

    tp = (prediction & target_bool).sum().item()
    tn = (~prediction & ~target_bool).sum().item()
    fp = (prediction & ~target_bool).sum().item()
    fn = (~prediction & target_bool).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

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

    print(f"Device: {device}")
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Test threshold: {threshold}"
    )

    dataset = LEVIRChangeDataset(
        split="test",
        size=256,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test samples: {len(dataset)}"
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

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Checkpoint validation IoU: "
        f"{checkpoint.get('val_iou'):.4f}"
    )

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for image_a, image_b, target in loader:

            image_a = image_a.to(device)
            image_b = image_b.to(device)
            target = target.to(device)

            logits = model(
                image_a,
                image_b,
            )

            metrics = calculate_metrics(
                logits,
                target,
                threshold,
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

    total_pixels = (
        total_tp
        + total_tn
        + total_fp
        + total_fn
    )

    results = {
        "checkpoint": checkpoint_path,
        "threshold": threshold,
        "test_samples": len(dataset),
        "precision": precision,
        "recall": recall,
        "dice_f1": dice,
        "iou": iou,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "total_pixels": total_pixels,
    }

    print()
    print("=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"Dice/F1   : {dice:.4f}"
    )

    print(
        f"IoU       : {iou:.4f}"
    )

    print("-" * 70)

    print(
        f"TP        : {total_tp}"
    )

    print(
        f"TN        : {total_tn}"
    )

    print(
        f"FP        : {total_fp}"
    )

    print(
        f"FN        : {total_fn}"
    )

    print("=" * 70)

    output_dir = Path(
        "outputs/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "change_detection_test_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "Results saved:",
        output_path,
    )


if __name__ == "__main__":
    main()