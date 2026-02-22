import subprocess
import urllib.parse
from pathlib import Path

def build_arcgis_query_url(layer_url, bbox_param):
    params = {
        "where": "1=1",
        "geometry": bbox_param,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25833",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "25833",
        "f": "geojson",
    }

    return layer_url.rstrip("/") + "/query?" + urllib.parse.urlencode(params)


def fetch_layer_for_region(bbox,region_gpkg, out_gpkg, layer_name, layer_url, geometry_type, region_layer="region", crs="EPSG:25833"):
    """
    Use ogr2ogr to fetch a single ArcGIS MapServer layer for a region
    the bash command for this would be:
    
    ogr2ogr -f GPKG -update "$OUT_GPKG" \
        'https://nve.geodataonline.no/arcgis/rest/services/
        Innsjodatabase2/MapServer/5/query?where=1%3D1&geometry=${BBOX_MINX}%2C${BBOX_MINY}%2C${BBOX_MAXX}%2C${BBOX_MAXY}&geometryType=esriGeometryEnvelope&inSR=25833&spatialRel=esriSpatialRelIntersects&outFields=*&returnGeometry=true&outSR=25833&f=geojson' \
        -nln lakes -nlt MULTIPOLYGON -t_srs EPSG:25833 \
        -clipsrc "$KOMM_GPKG" -clipsrclayer "$KOMM_LAYER"
    """
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    url = build_arcgis_query_url(layer_url, bbox)

    if out_gpkg.exists():
        action_flag = "-update"
    else:
        action_flag = "-overwrite"

    cmd = [
        "ogr2ogr",
        "-f", "GPKG",
        action_flag,
        str(out_gpkg),
        url,
        "-nln", layer_name,
        "-t_srs", crs,
        "-clipsrc", str(region_gpkg),
        "-clipsrclayer", region_layer,
        "-nlt", geometry_type,
    ]

    subprocess.run(cmd, check=True)
