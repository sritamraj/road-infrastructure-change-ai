from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_PATH = Path(
    "data/raw/deepglobe/train/images/142436_sat.jpg"
)

MASK_PATH = Path(
    "outputs/predictions/142436_pred.png"
)

SKELETON_PATH = Path(
    "outputs/predictions/142436_skeleton.png"
)

OUTPUT_PATH = Path(
    "outputs/predictions/142436_network_result.png"
)


image = np.array(
    Image.open(IMAGE_PATH).convert("RGB")
)

mask = np.array(
    Image.open(MASK_PATH).convert("L")
) > 127

skeleton = np.array(
    Image.open(SKELETON_PATH).convert("L")
) > 127


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(image)
axes[0].set_title("Satellite Image")
axes[0].axis("off")


axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Predicted Road Mask")
axes[1].axis("off")


axes[2].imshow(image)

y, x = np.where(skeleton)

axes[2].scatter(
    x,
    y,
    s=0.5,
)

axes[2].set_title(
    "Extracted Road Centerline"
)

axes[2].axis("off")


plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {OUTPUT_PATH}"
)