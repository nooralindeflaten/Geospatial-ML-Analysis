import subprocess
import numpy as np
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import box

def load_region(region_path: str, crs):
    region = gpd.read_file(region_path, layer="region")
    if len(region) != 1:
        region = region.dissolve(by=None)
    region = region.to_crs(crs)
    return region

def build_region_dtm(region_name: str, region_path: str, tile_index, dtm_dir: Path, crs="EPSG:25833", res=10):
    """
    For a given region:
      - find intersecting tiles from the index
      - mosaic them with gdalbuildvrt
      - clip mosaic with region polygon using gdalwarp -cutline
      - save <region>_dtm10.tif
    """
    print(f"\n=== {region_name}: building DTM10 from tiles ===")
    region = load_region(region_path, tile_index.crs)
    region_geom = region.geometry.iloc[0]

    # Find all tiles whose bbox intersects the region
    intersects = tile_index.intersects(region_geom.buffer(0))
    tiles_for_region = tile_index[intersects]

    if tiles_for_region.empty:
        print(f"  No tiles intersect region {region_name} - check CRS / extents.")
        return

    tile_paths = [str(p) for p in tiles_for_region["path"]]
    print(f"  Using {len(tile_paths)} tiles")

    dtm_dir.mkdir(parents=True, exist_ok=True)
    
    # remove files if exist from previous runs during debugging
    vrt_path = Path(dtm_dir / f"{region_name}_dtm10_tmp.vrt")
    out_tif = Path(dtm_dir / f"{region_name}_dtm10.tif")
    if vrt_path.exists():
        vrt_path.unlink()
    if out_tif.exists():
        out_tif.unlink()

    cmd_vrt = ["gdalbuildvrt", str(vrt_path)] + tile_paths
    print("     ", " ".join(cmd_vrt))
    subprocess.check_call(cmd_vrt)

    print("  -> Clipping with gdalwarp -cutline...")
    cmd_warp = [
        "gdalwarp",
        "-cutline", str(region_path),
        "-cl", "region",
        "-crop_to_cutline",
        "-t_srs", crs,
        "-of", "GTiff",
        "-tr", str(res), str(res),
        "-dstnodata", str(np.nan),
        str(vrt_path),
        str(out_tif),
    ]
    print("     ", " ".join(cmd_warp))
    subprocess.check_call(cmd_warp)
