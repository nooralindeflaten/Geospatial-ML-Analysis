import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def build_examples_for_region(gpkg,gpkg_layer, region_name,json_path=None,csv_path=None):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    region = gpd.read_file(gpkg, layer=gpkg_layer).to_crs("EPSG:25833")
    sample = region.head().copy()
    json_input = {
        "region": region_name,
        "source_path": str(gpkg),
        "source_layer": gpkg,
        "source_crs": "EPSG:25833",
        "features": []
    }
    one_sample = sample.iloc[0]
    for idx, row in one_sample.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        json_input["features"].append({
            "id": idx,
            "geometry": {
                "type": "Point",
                "coordinates": [centroid.x, centroid.y]
            },
            "properties": row.drop("geometry").to_dict()
        })
    with open(json_path, "w") as f:
        import json
        json.dump(json_input, f, indent=2)
    df = pd.DataFrame(sample.drop(columns="geometry"))
    df.to_csv(csv_path, index=False)
    print(f"Saved example JSON to {json_path} and CSV to {csv_path}")
    
    