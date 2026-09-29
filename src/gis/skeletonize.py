import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
    skeletonize,
)


def main():
    parser = argparse.ArgumentParser(
        description="Clean a road mask and extract its skeleton."
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
        "--min_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    mask = np.array(
        Image.open(args.input).convert("L")
    ) > 0

    print(
        f"Input road pixels: {int(mask.sum())}"
    )

    cleaned = remove_small_objects(
        mask,
        min_size=args.min_size,
    )

    footprint = disk(args.radius)

    cleaned = binary_opening(
        cleaned,
        footprint,
    )

    cleaned = binary_closing(
        cleaned,
        footprint,
    )

    skeleton = skeletonize(cleaned)

    print(
        f"Cleaned road pixels: "
        f"{int(cleaned.sum())}"
    )

    print(
        f"Skeleton pixels: "
        f"{int(skeleton.sum())}"
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        (skeleton.astype(np.uint8) * 255)
    ).save(output)

    print(
        f"Saved skeleton: {output}"
    )


if __name__ == "__main__":
    main()