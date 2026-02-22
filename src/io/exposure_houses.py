from pathlib import Path
import geopandas as gpd
import urllib.parse


def fetch_bygningspunkt_for_region(region_gpkg,region_name, region_layer="region", out_path=None):
    bygningspunkt_wfs_url = "https://wfs.geonorge.no/skwms1/wfs.matrikkelen-bygningspunkt"
    bygningspunkt_layer = "Bygning"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    region = gpd.read_file(region_gpkg, layer=region_layer).to_crs("EPSG:25833")
    if len(region) != 1:
        region = region.dissolve(by=None)
    geom = region.geometry.iloc[0]

    
    minx, miny, maxx, maxy = geom.bounds

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typename": bygningspunkt_layer,
        "srsName": "EPSG:25833",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:25833",
    }

    url_with_params = bygningspunkt_wfs_url + "?" + urllib.parse.urlencode(params)
    gdf = gpd.read_file(url_with_params)

    gdf = gdf.to_crs("EPSG:25833")

    gdf = gpd.overlay(gdf, region, how="intersection")

    gdf.to_file(out_path, driver="GPKG")
    print(f"Saved {len(gdf)} bygningspunkt to {out_path}")
