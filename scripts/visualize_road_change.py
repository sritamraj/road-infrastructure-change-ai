from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


T1_PATH = Path("outputs/predictions/142436_pred.png")
T2_PATH = Path("outputs/predictions/142436_t2_synthetic.png")
CHANGE_PATH = Path("outputs/predictions/142436_road_change.png")

OUTPUT_PATH = Path(
    "outputs/predictions/142436_road_change_result.png"
)


def load_mask(path):
    return np.array(
        Image.open(path).convert("L")
    ) > 0


def main():
    t1 = load_mask(T1_PATH)
    t2 = load_mask(T2_PATH)
    change = load_mask(CHANGE_PATH)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(t1, cmap="gray")
    axes[0].set_title("T1 Road Mask")
    axes[0].axis("off")

    axes[1].imshow(t2, cmap="gray")
    axes[1].set_title("T2 Road Mask")
    axes[1].axis("off")

    axes[2].imshow(change, cmap="gray")
    axes[2].set_title("Detected Road Change")
    axes[2].axis("off")

    fig.suptitle(
        "Temporal Road-Network Change Analysis",
        fontsize=14,
    )

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

    print(f"Saved visualization: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()