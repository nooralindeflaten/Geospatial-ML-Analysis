import rasterio
from pathlib import Path
from shapely.geometry import box
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import subprocess
import richdem as rd 
import urllib.parse


def build_wcs_getcoverage_url(minx, miny, maxx, maxy, res=10.0):
    """
    Build a WCS GetCoverage URL for your DTM10 service.
    """
    dtm_base = "http://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
    dtm_coverage_id = "nhm_dtm_topo_25833"  # This is the coverage name for DTM10 in the WCS service
    crs = "EPSG:25833"  # UTM zone 33N, which covers Norway
    params = {
        "SERVICE": "WCS",
        "REQUEST": "GetCoverage",
        "VERSION": "1.0.0",
        "COVERAGE": dtm_coverage_id,   # sometimes COVERAGEID in 2.0.1
        "CRS": crs,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "RESX": str(res),
        "RESY": str(res),
        "FORMAT": "GeoTIFF",
    }
    # For GDAL WCS, you’ll usually use the "WCS:" prefix in the datasource
    query = urllib.parse.urlencode(params)
    return f"WCS:{dtm_base}?{query}"


def fetch_dtm_for_region_wcs(bbox,region_name, out_tif, patch_res_m=10.0):

    minx, miny, maxx, maxy = bbox[0], bbox[1], bbox[2], bbox[3]

    subset_url = (
        "http://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
        "?VERSION=1.0.0&COVERAGE=nhm_dtm_topo_25833"
    )
    print(f"\n=== {region_name}: fetching DTM10 ===")

    # 1) gdal_translate: WCS → bbox raster
    # 2) gdalwarp with cutline: clip to exact region polygon
    if out_tif.exists():
        out_tif.unlink()
    crs = "EPSG:25833"
    cmd = [
        "gdal_translate",
        "-projwin",
        str(minx), str(maxy), str(maxx), str(miny),  # ULX ULY LRX LRY
        "-projwin_srs", crs,
        "-tr", str(patch_res_m), str(patch_res_m),
        f"WCS:{subset_url}",
        str(out_tif),
        ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] gdal_translate failed for house {region_name}: {e}")
        return None

    return out_tif
