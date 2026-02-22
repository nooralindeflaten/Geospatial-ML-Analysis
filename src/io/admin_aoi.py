import pandas as pd
import geopandas as gpd
import requests
import urllib.parse
from pathlib import Path

def admin_aoi(out_path):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typename": "app:Kommune", # This layer name was found by inspecting the WFS capabilities document at https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?service=WFS&version=2.0.0&request=GetCapabilities
        "srsName": "EPSG:25833",
    }
    admin_wfs = "https://wfs.geonorge.no/skwms1/wfs.administrative_enheter"
    url_with_params = admin_wfs + "?" + urllib.parse.urlencode(params)
    gdf = gpd.read_file(url_with_params)
    gdf = gdf.to_crs("EPSG:25833")
    out_path.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path / "admin.gpkg", layer="app:Kommune", driver="GPKG")
    print(f"Saved administrative units to {out_path / 'admin.gpkg'}")

