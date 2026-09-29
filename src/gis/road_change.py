import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_mask(path):
    mask = np.array(
        Image.open(path).convert("L")
    )

    return mask > 0


def save_mask(mask, path):
    output = (mask.astype(np.uint8) * 255)

    Image.fromarray(output).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two aligned road masks."
    )

    parser.add_argument(
        "--t1",
        required=True,
        help="T1 road mask."
    )

    parser.add_argument(
        "--t2",
        required=True,
        help="T2 road mask."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output change mask."
    )

    parser.add_argument(
        "--min_size",
        type=int,
        default=20,
        help="Minimum connected change component size."
    )

    args = parser.parse_args()

    t1 = load_mask(args.t1)
    t2 = load_mask(args.t2)

    if t1.shape != t2.shape:
        raise ValueError(
            f"T1 and T2 masks must have the same shape. "
            f"Got {t1.shape} and {t2.shape}."
        )

    # Newly appearing road pixels
    added = t2 & ~t1

    # Road pixels present in T1 but absent in T2
    removed = t1 & ~t2

    # Any road-state change
    changed = added | removed

    # Remove very small isolated regions
    labeled, num = ndimage.label(changed)

    cleaned = np.zeros_like(changed)

    for component_id in range(1, num + 1):
        component = labeled == component_id

        if component.sum() >= args.min_size:
            cleaned |= component

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_mask(cleaned, output)

    print("=" * 60)
    print("ROAD CHANGE ANALYSIS")
    print("=" * 60)

    print(f"T1 road pixels: {int(t1.sum())}")
    print(f"T2 road pixels: {int(t2.sum())}")

    print(
        f"Added road pixels: {int(added.sum())}"
    )

    print(
        f"Removed road pixels: {int(removed.sum())}"
    )

    print(
        f"Changed road pixels before filtering: "
        f"{int(changed.sum())}"
    )

    print(
        f"Changed road pixels after filtering: "
        f"{int(cleaned.sum())}"
    )

    print(f"Output: {output}")

    print("=" * 60)


if __name__ == "__main__":
    main()