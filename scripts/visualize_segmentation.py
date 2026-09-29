from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_PATH = Path("data/raw/deepglobe/train/images/142436_sat.jpg")
GT_PATH = Path("data/raw/deepglobe/train/masks/142436_mask.png")
PRED_PATH = Path("outputs/predictions/142436_pred.png")
OUTPUT_PATH = Path("outputs/predictions/142436_segmentation_result.png")


image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

gt_rgb = np.array(
    Image.open(GT_PATH).convert("RGB")
)

gt = np.all(
    gt_rgb == 255,
    axis=2
)

pred = np.array(
    Image.open(PRED_PATH).convert("L")
) > 127


fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)


axes[0].imshow(image)
axes[0].set_title("Satellite Image")
axes[0].axis("off")


axes[1].imshow(gt, cmap="gray")
axes[1].set_title("Ground Truth Road Mask")
axes[1].axis("off")


axes[2].imshow(pred, cmap="gray")
axes[2].set_title("U-Net Prediction")
axes[2].axis("off")


overlay = image.copy()

overlay[pred] = (
    0.5 * overlay[pred]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

axes[3].imshow(overlay)
axes[3].set_title("Prediction Overlay")
axes[3].axis("off")


plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {OUTPUT_PATH}")