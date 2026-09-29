import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


def main():
    parser = argparse.ArgumentParser(
        description="Convert a skeleton road mask into pixel-space line features."
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--min_length",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    skeleton = np.array(
        Image.open(args.input).convert("L")
    ) > 0

    skeleton = skeletonize(skeleton)

    labeled = label(
        skeleton,
        connectivity=2,
    )

    lines = []

    for region in regionprops(labeled):
        if region.area < args.min_length:
            continue

        coords = region.coords

        if len(coords) < 2:
            continue

        points = [
            (float(col), float(row))
            for row, col in coords
        ]

        line = LineString(points)

        if line.length >= args.min_length:
            lines.append(line)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdf = gpd.GeoDataFrame(
        {
            "class": ["road"] * len(lines),
            "length_px": [
                float(line.length)
                for line in lines
            ],
        },
        geometry=lines,
        crs=None,
    )

    gdf.to_file(
        output,
        driver="GeoJSON",
    )

    print(
        f"Skeleton pixels: {int(skeleton.sum())}"
    )

    print(
        f"Line features: {len(gdf)}"
    )

    print(
        f"Output: {output}"
    )

    if not gdf.empty:
        print(
            f"Total network length (pixel units): "
            f"{float(gdf.length.sum()):.2f}"
        )


if __name__ == "__main__":
    main()