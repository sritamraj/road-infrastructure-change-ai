import argparse

import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pixel-space road-network statistics."
    )

    parser.add_argument(
        "--roads",
        required=True,
        help="Input road-network GeoJSON."
    )

    args = parser.parse_args()

    g = gpd.read_file(args.roads)

    # The current road network uses raster pixel coordinates,
    # not real geographic coordinates. GeoJSON readers may assign
    # EPSG:4326 automatically, so remove that interpretation here.
    if g.crs is not None:
        g = g.set_crs(None, allow_override=True)

    print("=" * 60)
    print("ROAD NETWORK STATISTICS")
    print("=" * 60)

    print(f"Number of road features: {len(g)}")

    if g.empty:
        print("No road features found.")
        print("=" * 60)
        return

    lengths = g.geometry.length

    total_length = float(lengths.sum())
    mean_length = float(lengths.mean())
    min_length = float(lengths.min())
    max_length = float(lengths.max())

    minx, miny, maxx, maxy = g.total_bounds

    width = maxx - minx
    height = maxy - miny
    network_extent_area = width * height

    print("Coordinate system: pixel coordinates")
    print()

    print(f"Total network length: {total_length:.2f} pixels")
    print(f"Average feature length: {mean_length:.2f} pixels")
    print(f"Shortest feature: {min_length:.2f} pixels")
    print(f"Longest feature: {max_length:.2f} pixels")

    print()

    print(
        f"Network bounding box: "
        f"{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}"
    )

    print(
        f"Network bounding-box width: "
        f"{width:.2f} pixels"
    )

    print(
        f"Network bounding-box height: "
        f"{height:.2f} pixels"
    )

    print(
        f"Network bounding-box area: "
        f"{network_extent_area:.2f} square pixels"
    )

    print()

    print("NOTE:")
    print("These measurements are in raster pixel units.")
    print("They are not physical distances in meters or kilometers.")
    print("=" * 60)


if __name__ == "__main__":
    main()