import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


def main():
    parser = argparse.ArgumentParser(
        description="Extract road-change hotspots from a binary change mask."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Binary road-change mask."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="CSV output file."
    )

    parser.add_argument(
        "--min_size",
        type=int,
        default=20,
        help="Minimum hotspot size in pixels."
    )

    args = parser.parse_args()

    mask = np.array(
        Image.open(args.input).convert("L")
    ) > 0

    labeled = label(
        mask,
        connectivity=2,
    )

    regions = []

    for region in regionprops(labeled):
        area = int(region.area)

        if area < args.min_size:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        regions.append(
            {
                "hotspot_id": len(regions) + 1,
                "area_pixels": area,
                "centroid_x": float(region.centroid[1]),
                "centroid_y": float(region.centroid[0]),
                "min_x": int(min_col),
                "min_y": int(min_row),
                "max_x": int(max_col),
                "max_y": int(max_row),
                "width_pixels": int(max_col - min_col),
                "height_pixels": int(max_row - min_row),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", encoding="utf-8") as f:
        f.write(
            "hotspot_id,area_pixels,centroid_x,centroid_y,"
            "min_x,min_y,max_x,max_y,width_pixels,height_pixels\n"
        )

        for r in regions:
            f.write(
                f"{r['hotspot_id']},"
                f"{r['area_pixels']},"
                f"{r['centroid_x']:.2f},"
                f"{r['centroid_y']:.2f},"
                f"{r['min_x']},"
                f"{r['min_y']},"
                f"{r['max_x']},"
                f"{r['max_y']},"
                f"{r['width_pixels']},"
                f"{r['height_pixels']}\n"
            )

    print("=" * 60)
    print("ROAD-CHANGE HOTSPOT ANALYSIS")
    print("=" * 60)

    print(f"Input: {args.input}")
    print(f"Hotspots detected: {len(regions)}")
    print(f"Output: {output}")

    if regions:
        print()

        for r in regions:
            print(
                f"Hotspot {r['hotspot_id']}: "
                f"area={r['area_pixels']} px, "
                f"centroid=({r['centroid_x']:.2f}, "
                f"{r['centroid_y']:.2f}), "
                f"size={r['width_pixels']}x"
                f"{r['height_pixels']} px"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()