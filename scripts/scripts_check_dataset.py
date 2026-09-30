from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PROJECT = Path("data/raw/deepglobe/train")
OUTPUT = Path("outputs/dataset_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

image_files = sorted((PROJECT / "images").glob("*_sat.jpg"))
if not image_files:
    raise SystemExit(f"DeepGlobe dataset not found. Expected images under: {PROJECT / "images"}")
image_path = image_files[0]

sample_id = image_path.stem.replace("_sat", "")
mask_path = PROJECT / "masks" / f"{sample_id}_mask.png"
if not mask_path.exists():
    raise SystemExit(f"Mask not found for sample {sample_id}: {mask_path}")

image = np.array(Image.open(image_path).convert("RGB"))
mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

road_mask = np.all(mask_rgb == 255, axis=2)

overlay = image.copy()
overlay[road_mask] = [255, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
axes[0].set_title(f"Satellite\n{sample_id}")
axes[0].axis("off")

axes[1].imshow(road_mask, cmap="gray")
axes[1].set_title("Ground Truth Road Mask")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Road Overlay")
axes[2].axis("off")

plt.tight_layout()

output_path = OUTPUT / "sample_road_pair.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Image: {image_path}")
print(f"Mask:  {mask_path}")
print(f"Output: {output_path}")
print(f"Image shape: {image.shape}")
print(f"Mask shape: {road_mask.shape}")
print(f"Road pixels: {road_mask.sum()}")
print(f"Road percentage: {road_mask.mean() * 100:.2f}%")