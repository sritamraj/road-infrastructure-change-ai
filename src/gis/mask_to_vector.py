import argparse
from pathlib import Path
import rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mask',required=True);ap.add_argument('--output',required=True);ap.add_argument('--min_area',type=float,default=0);a=ap.parse_args()
    with rasterio.open(a.mask) as s: arr=s.read(1); transform=s.transform; crs=s.crs
    if crs is None: raise ValueError('Input raster has no CRS.')
    geoms=[shape(g) for g,v in shapes(arr.astype('uint8'),mask=arr>0,transform=transform) if shape(g).area>=a.min_area]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);gpd.GeoDataFrame({'class':['road']*len(geoms)},geometry=geoms,crs=crs).to_file(out,driver='GeoJSON');print(f'Wrote {len(geoms)} features to {out}')
if __name__=='__main__':main()
