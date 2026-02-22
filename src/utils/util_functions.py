import geopandas as gpd


def get_region_bbox(region_gpkg, region_layer="region", crs="EPSG:25833"):
    region = gpd.read_file(region_gpkg, layer=region_layer)
    if len(region) != 1:
        region = region.dissolve(by=None)
    region = region.to_crs(crs)
    minx, miny, maxx, maxy = region.total_bounds
    return minx, miny, maxx, maxy
