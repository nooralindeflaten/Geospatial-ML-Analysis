from pathlib import Path
from src.io.arcgis_fetch import fetch_layer_for_region
from src.io.examples_builder import build_examples_for_region
import pandas as pd
import geopandas as gpd

HYDRO_LAYERS = [
    {
        "name": "vannforing_stasjoner",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/HydrologiskeData3/MapServer/0/",
        "geometry_type": "POINT",  
    },
    {
        "name": "rivers",
        "url": "https://kart.nve.no/enterprise/rest/services/Elvenett1/MapServer/2/",
        "geometry_type": "MULTILINESTRING",  
    },
    {
        "name": "lakes",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/Innsjodatabase2/MapServer/5/",
        "geometry_type": "MULTIPOLYGON",  
    },
    {
        "name": "catchments",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/Nedborfelt2/MapServer/1/",
        "geometry_type": "MULTIPOLYGON",  
    },
]

def build_all_region_hydro(bbox,region_name,region_gpkg, hydro_path):
    out_gpkg = Path(hydro_path)

    if out_gpkg.exists():
        print(f"[{region_name}] Removing existing {out_gpkg}")
        out_gpkg.unlink()

    for layer_cfg in HYDRO_LAYERS:
        fetch_layer_for_region(
            bbox,
            region_gpkg,
            out_gpkg,
            layer_name=layer_cfg["name"],
            layer_url=layer_cfg["url"],
            geometry_type=layer_cfg["geometry_type"],
        )
        # print(f"{layer_cfg['name']} done") <- debug line
    
    print(f"✓ Added hydro layers for {region_name},  at {out_gpkg} \n")
