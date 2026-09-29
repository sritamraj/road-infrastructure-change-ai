from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.change_detection.dataset import LEVIRChangeDataset
from src.change_detection.models import SiameseChangeNet


CHECKPOINT = Path("outputs/checkpoints/siamese_change_best.pt")
OUTPUT_DIR = Path("outputs/predictions/change_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

dataset = LEVIRChangeDataset(
    split="val",
    size=256,
    train=False,
)

print("Validation samples:", len(dataset))

model = SiameseChangeNet().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])

elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

else:
    model.load_state_dict(checkpoint)

model.eval()

print("Checkpoint loaded successfully.")

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])

    if "val_iou" in checkpoint:
        print("Checkpoint validation IoU:", checkpoint["val_iou"])


for index in [0, 1, 2, 3, 4]:

    image_a, image_b, label = dataset[index]

    with torch.no_grad():

        logits = model(
            image_a.unsqueeze(0).to(device),
            image_b.unsqueeze(0).to(device),
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    target = label[0].numpy()

    print(f"\nSample {index}")

    print(
        "Probability min:",
        float(probability.min())
    )

    print(
        "Probability max:",
        float(probability.max())
    )

    print(
        "Probability mean:",
        float(probability.mean())
    )

    print(
        "Probability median:",
        float(np.median(probability))
    )

    print(
        "Ground-truth changed pixels:",
        int(target.sum())
    )

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:

        predicted = probability >= threshold

        print(
            f"Threshold {threshold:.1f}: "
            f"predicted changed pixels = "
            f"{int(predicted.sum())}"
        )

    probability_image = (
        probability * 255
    ).clip(0, 255).astype(np.uint8)

    output_path = (
        OUTPUT_DIR /
        f"sample_{index}_probability.png"
    )

    Image.fromarray(
        probability_image
    ).save(output_path)

    print("Saved:", output_path)


print(
    "\nSaved probability maps to:",
    OUTPUT_DIR
)