from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self): return self.root / "data"
    @property
    def raw(self): return self.data / "raw"
    @property
    def processed(self): return self.data / "processed"
    @property
    def cache(self): return self.data / "cache"
    @property
    def downloads(self): return self.data / "downloads"
    
    def dtm_tiles(self): return self.raw / "rasters" / "dtm10" / "tiles"

    def region_dir(self, region: str) -> Path:
        return self.processed / "regions" / region

@dataclass
class GlobalContext:
    paths: ProjectPaths
    
    @property
    def tile_index_path(self):
        return self.paths.cache / "dtm_tile_index.gpkg"
    
    @property
    def admin_aoi_path(self):
        return self.paths.raw / "vector" / "admin" / "admin.gpkg"

@dataclass
class RegionContext:
    name: str
    paths: ProjectPaths

    @property
    def aoi_gpkg(self):
        return self.paths.raw / "vector" / "admin" / f"{self.name}_area.gpkg"
    
   
    @property
    def dtm_wcs_path(self):
        return self.paths.raw / "rasters" / "dtm10" / "wcs" / f"{self.name}_dtm10.tif"
    @property
    def dtm_tile_path(self):
        return self.paths.dtm_tiles() / f"{self.name}_dtm10.tif"
    @property
    def houses_gpkg(self):
        return self.paths.raw / "vector" / "exposure" / f"{self.name}_houses.gpkg"
    @property
    def hydro_gpkg(self):
        return self.paths.raw / "vector" / "hydro" / f"{self.name}_hydro.gpkg"
    @property
    def hazard_gpkg(self):
        return self.paths.raw / "vector" / "hazards" / f"{self.name}_hazards.gpkg"
    @property
    def nve_stations_active_path(self):
        return self.paths.raw / "vector" / "stations" / f"{self.name}_nve_active_st.json"
    @property
    def defnve_stations_inactive_path(self):
        return self.paths.raw / "vector" / "stations" / f"{self.name}_nve_inactive_st.json"
    @property
    def frost_stations_path(self):
        return self.paths.raw / "vector" / "stations" / f"{self.name}_frost_stations.json"
    @property
    def house_layer(self):
        return f"{self.name}_houses"
    @property
    def region_dir(self):
        return self.paths.region_dir(self.name)
    @property
    def patch_index_parquet(self):
        return self.region_dir / "ml" / f"{self.name}_patch_index.parquet"
    
    def rasters_dir(self):
        return self.region_dir / "rasters"
    def vectors_dir(self):
        return self.region_dir / "vectors"
    def hydro_dir(self):
        return self.region_dir / "hydro"
    def hazard_dir(self):
        return self.region_dir / "hazard"
    def ml_dir(self):
        return self.region_dir / "ml"
    def link_dir(self):
        return self.region_dir / "links"
    def graphs_dir(self):
        return self.region_dir / "graphs"
    def raster_paths(self):
        return {
            "dtm": self.rasters_dir() / f"{self.name}_dtm10.tif",
            "slope": self.rasters_dir() / f"{self.name}_slope_deg.tif",
            "aspect": self.rasters_dir() / f"{self.name}_aspect_deg.tif",
            "curv": self.rasters_dir() / f"{self.name}_curvature.tif",
            "flowacc": self.rasters_dir() / f"{self.name}_flowacc_d8.tif",
            "twi": self.rasters_dir() / f"{self.name}_twi.tif",
        }
    def flood_links_path(self):
        return self.link_dir() / f"{self.name}_flood_links.parquet"
    def hydro_links_path(self):
        return self.link_dir() / f"{self.name}_hydro_links.parquet"
    def landslide_links_path(self):
        return self.link_dir() / f"{self.name}_landslide_links.parquet"


@dataclass
class RasterPaths:
    region: RegionContext
    name: str
    
    @property
    def dtm(self):
        return self.region.dtm_tile_path
    @property
    def slope(self):
        return self.region.rasters_dir() / f"{self.region.name}_slope_deg.tif"
    @property
    def aspect(self):
        return self.region.rasters_dir() / f"{self.region.name}_aspect_deg.tif"
    @property
    def curvature(self):
        return self.region.rasters_dir() / f"{self.region.name}_curvature.tif"
    @property
    def flowacc(self):
        return self.region.rasters_dir() / f"{self.region.name}_flowacc_d8.tif"
    @property
    def twi(self):
        return self.region.rasters_dir() / f"{self.region.name}_twi.tif"
    
    def as_dict(self):
        return {
            "dtm": self.dtm,
            "slope": self.slope,
            "aspect": self.aspect,
            "curvature": self.curvature,
            "flowacc": self.flowacc,
            "twi": self.twi,
        }
    
    def as_list(self):
        return list(self.as_dict().values())