import geopandas as gpd
import folium


INPUT = (
    "outputs/predictions/"
    "142436_road_network.geojson"
)

OUTPUT = (
    "outputs/predictions/"
    "142436_road_network_map.html"
)


gdf = gpd.read_file(INPUT)

if gdf.empty:
    raise RuntimeError(
        "The road network GeoJSON is empty."
    )


minx, miny, maxx, maxy = gdf.total_bounds

center_x = (minx + maxx) / 2
center_y = (miny + maxy) / 2


m = folium.Map(
    location=[center_y, center_x],
    zoom_start=12,
    tiles=None,
)


folium.GeoJson(
    gdf.to_json(),
    name="Extracted Road Network",
    tooltip=folium.GeoJsonTooltip(
        fields=["class", "length_px"],
        aliases=[
            "Class",
            "Length (pixels)",
        ],
    ),
).add_to(m)


folium.LayerControl().add_to(m)

m.save(OUTPUT)

print(
    f"Saved interactive map: {OUTPUT}"
)