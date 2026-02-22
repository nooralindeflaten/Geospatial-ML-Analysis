import subprocess
import numpy as np
from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import box
import sys
import os
from src.io.admin_aoi import admin_aoi
from src.utils.paths import ProjectPaths, GlobalContext, RegionContext
from src.pipeline.run_region import RegionBootstrapper

TILES_DIR = Path("geodata_analysis_final_version/data/downloads/Nedlastingspakke/")
def build_tile_index():
    tifs = sorted(TILES_DIR.glob("*_10m_z33.tif"))
    if not tifs:
        raise SystemExit(f"No .tif tiles found in {TILES_DIR}")

    records = []
    crs = 25833

    for p in tifs:
        with rasterio.open(p) as src:
            bounds = src.bounds
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            records.append({"path": str(p), "geometry": geom})

    gdf = gpd.GeoDataFrame(records, crs=25833)
    print(f"Indexed {len(gdf)} tiles, CRS={gdf.crs}")
    return gdf



def delete_existing_outputs(region_ctx: RegionContext): 
    paths_to_check = [
        region_ctx.aoi_gpkg,
        region_ctx.houses_gpkg,
        region_ctx.hydro_gpkg,
        region_ctx.hazard_gpkg,
        region_ctx.dtm_wcs_path,
        region_ctx.dtm_tile_path,
    ]
    for p in paths_to_check:
        if p.exists():
            print(f"  Deleting existing file {p} from previous run...")
            p.unlink()
            
            
def closest_landslide_links(region_ctx: RegionContext, houses_gdf, hazard_gpkg, out_path):
    landslide_df = gpd.read_file(hazard_gpkg, layer="landslides").to_crs(25833)
    houses_gdf["distance_m"] = houses_gdf.apply(lambda row: landslide_df.geometry.distance(row.geometry).min(), axis=1)
    houses_gdf[["bygningsnummer", "distance_m"]].to_parquet(out_path, index=False)
def main(regions):
    tile_index = build_tile_index()
    proj_paths = ProjectPaths(root=Path("geodata_analysis_final_version"))
    global_ctx = GlobalContext(paths=proj_paths)
    admin_gpkg = global_ctx.admin_aoi_path
    for region_name, kommune_names in regions.items():
        region_ctx = RegionContext(name=region_name, paths=proj_paths)
        bootstrapper = RegionBootstrapper(proj_paths, region_ctx, kommune_names, admin_gpkg, tile_index)
        bootstrapper.run_links()     
    
if __name__ == "__main__":

    print("=== Initiating root data ===")
    regions = {
        "sogn": ["Årdal","Lærdal","Luster"]
    }
    main(regions)