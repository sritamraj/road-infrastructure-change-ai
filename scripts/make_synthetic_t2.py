from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


input_path = Path("outputs/predictions/142436_pred.png")
output_path = Path("outputs/predictions/142436_t2_synthetic.png")

image = Image.open(input_path).convert("L")

draw = ImageDraw.Draw(image)

# Add a controlled road-like region to simulate a T2 change.
draw.rectangle(
    (700, 150, 850, 190),
    fill=255,
)

image.save(output_path)

print(f"T1 input: {input_path}")
print(f"Synthetic T2 output: {output_path}")
print("Synthetic road addition created successfully.")