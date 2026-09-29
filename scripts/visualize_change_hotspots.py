from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


CHANGE_PATH = Path(
    "outputs/predictions/142436_road_change.png"
)

CSV_PATH = Path(
    "outputs/metrics/142436_change_hotspots.csv"
)

OUTPUT_PATH = Path(
    "outputs/predictions/142436_change_hotspots.png"
)


def main():
    change = np.array(
        Image.open(CHANGE_PATH).convert("L")
    ) > 0

    rows = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

        for line in f:
            values = line.strip().split(",")

            if not values or len(values) != len(header):
                continue

            row = dict(zip(header, values))
            rows.append(row)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.imshow(
        change,
        cmap="gray",
    )

    for row in rows:
        hotspot_id = int(row["hotspot_id"])

        centroid_x = float(row["centroid_x"])
        centroid_y = float(row["centroid_y"])

        min_x = int(row["min_x"])
        min_y = int(row["min_y"])

        width = int(row["width_pixels"])
        height = int(row["height_pixels"])

        rectangle = patches.Rectangle(
            (min_x, min_y),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        ax.scatter(
            centroid_x,
            centroid_y,
            s=60,
            marker="x",
        )

        ax.text(
            centroid_x + 8,
            centroid_y - 8,
            f"Hotspot {hotspot_id}",
            fontsize=10,
        )

    ax.set_title(
        "Detected Road-Change Hotspots"
    )

    ax.set_xlabel(
        "Pixel X"
    )

    ax.set_ylabel(
        "Pixel Y"
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

    print(
        f"Saved hotspot visualization: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()