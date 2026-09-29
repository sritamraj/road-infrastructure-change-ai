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

    print(f"Device: {device}")

    checkpoint_path = (
        "outputs/checkpoints/"
        "siamese_change_weighted.pt"
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    dataset = LEVIRChangeDataset(
        split="val",
        size=256,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
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

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch"),
    )

    print(
        "Checkpoint validation IoU:",
        checkpoint.get("val_iou"),
    )

    print()

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    totals = {}

    for threshold in thresholds:

        totals[threshold] = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    with torch.no_grad():

        for image_a, image_b, target in loader:

            image_a = image_a.to(device)
            image_b = image_b.to(device)
            target = target.to(device)

            logits = model(
                image_a,
                image_b,
            )

            for threshold in thresholds:

                metrics = calculate_metrics(
                    logits,
                    target,
                    threshold,
                )

                totals[threshold]["tp"] += metrics["tp"]
                totals[threshold]["tn"] += metrics["tn"]
                totals[threshold]["fp"] += metrics["fp"]
                totals[threshold]["fn"] += metrics["fn"]

    print(
        "Validation threshold analysis "
        "(weighted model)"
    )

    print("=" * 78)

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'Dice/F1':>14}"
        f"{'IoU':>14}"
    )

    print("-" * 78)

    best_threshold = None
    best_iou = -1.0

    for threshold in thresholds:

        tp = totals[threshold]["tp"]
        tn = totals[threshold]["tn"]
        fp = totals[threshold]["fp"]
        fn = totals[threshold]["fn"]

        precision = (
            tp
            / (tp + fp + 1e-8)
        )

        recall = (
            tp
            / (tp + fn + 1e-8)
        )

        dice = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + 1e-8
            )
        )

        iou = (
            tp
            / (
                tp
                + fp
                + fn
                + 1e-8
            )
        )

        print(
            f"{threshold:10.2f}"
            f"{precision:14.4f}"
            f"{recall:14.4f}"
            f"{dice:14.4f}"
            f"{iou:14.4f}"
        )

        if iou > best_iou:

            best_iou = iou
            best_threshold = threshold

    print("=" * 78)

    print()
    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best validation IoU: "
        f"{best_iou:.4f}"
    )


if __name__ == "__main__":
    main()