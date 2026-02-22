# src/processing/aoi/bounds.py
from dataclasses import dataclass
import geopandas as gpd
from pathlib import Path

@dataclass(frozen=True)
class BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_arcgis_envelope(self) -> str:
        # used by arcgis_fetch.build_arcgis_query_url(..., bbox_param=...)
        return f"{self.minx},{self.miny},{self.maxx},{self.maxy}"

    def as_wfs_bbox(self, crs: str = "EPSG:25833") -> str:
        # used by WFS GetFeature bbox=...
        return f"{self.minx},{self.miny},{self.maxx},{self.maxy},{crs}"



def bbox_from_region(region_gpkg: Path, region_layer: str = "region", crs: str = "EPSG:25833") -> BBox:
    region = gpd.read_file(region_gpkg, layer=region_layer).to_crs(crs)
    if len(region) != 1:
        region = region.dissolve(by=None)
    minx, miny, maxx, maxy = region.total_bounds
    return BBox(minx, miny, maxx, maxy)
