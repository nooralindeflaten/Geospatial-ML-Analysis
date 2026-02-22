from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pandas as pd
import geopandas as gpd
from src.utils.paths import ProjectPaths,RegionContext, GlobalContext, RasterPaths
from dataclasses import dataclass


@dataclass
class Houses:
    region: str
    region_ctx: RegionContext
    
    @property
    def house_layer(self):
        return f"{self.region}_houses"
    @property
    def id_col(self):
        return "bygningsnummer"
    @property
    def houses_gdf(self):
        return gpd.read_file(self.region_ctx.houses_gpkg, layer=self.house_layer)
    @property
    def flood_links(self):
        return pd.read_parquet(self.region_ctx.flood_links_path())
    @property
    def hydro_links(self):
        return pd.read_parquet(self.region_ctx.hydro_links_path())
    @property
    def landslide_links(self):
        return pd.read_parquet(self.region_ctx.landslide_links_path())
    
    def get_links(self):
        return {
            "flood": self.flood_links,
            "hydro": self.hydro_links,
            "landslide": self.landslide_links,
        }
    
    def patch_size_m(self):
        return 200.0
    

    def tabular_cols(self):
        flood_cols =list(self.flood_links.columns)
        landslide_cols = list(self.landslide_links.columns)
        hydro_cols = list(self.hydro_links.columns)
        feature_cols = list(set(flood_cols + landslide_cols + hydro_cols))
        feature_cols.remove(self.id_col)
        return feature_cols    