
import geopandas as gpd
from src import processing,io,pipeline,utils
from src.processing.aoi.bounds import bbox_from_region
from src.io.admin_aoi import admin_aoi
from src.processing.aoi.aoi_builder import build_region_from_kommuner
from src.io.exposure_houses import fetch_bygningspunkt_for_region
from src.pipeline.hydro_pipeline import build_all_region_hydro
from src.pipeline.hazard_pipeline import build_all_region_hazard
from src.processing.terrain.derivatives import build_terrain_for_region, build_aspect_curve, processed_dtm
from src.processing.terrain.dtm_from_tiles import build_region_dtm
from src.utils.paths import ProjectPaths, RegionContext, GlobalContext
from src.io.wcs_dtm import fetch_dtm_for_region_wcs
from src.processing.links import landslide_links, flood_labels, hydro_links

class RegionBootstrapper:
    def __init__(self, proj_paths: ProjectPaths, region_ctx: RegionContext, kommune_names, admin_gpkg, tile_index):
        self.proj = proj_paths
        self.ctx = region_ctx
        self.kommune_names = kommune_names
        self.admin_gpkg = admin_gpkg
        self.tile_index = tile_index
        

    def build_region_aoi(self):
        build_region_from_kommuner(self.ctx.name, self.kommune_names, self.admin_gpkg, self.ctx.aoi_gpkg)

    def fetch_houses(self):
        fetch_bygningspunkt_for_region(self.ctx.aoi_gpkg,self.ctx.name,region_layer="region", out_path=self.ctx.houses_gpkg)

    def fetch_hydro_and_hazard(self):
        bbox = bbox_from_region(self.ctx.aoi_gpkg)
        build_all_region_hydro(bbox.as_arcgis_envelope(), self.ctx.name, self.ctx.aoi_gpkg, self.ctx.hydro_gpkg)
        build_all_region_hazard(bbox.as_arcgis_envelope(), self.ctx.name, self.ctx.aoi_gpkg, self.ctx.hazard_gpkg)

    def fetch_dtm(self):
        bbox = bbox_from_region(self.ctx.aoi_gpkg)
        fetch_dtm_for_region_wcs(bbox.as_wfs_bbox(), self.ctx.name, self.ctx.dtm_wcs_path)
    
    def clip_dtm_to_region(self):
        dtm_tiles_dir = self.proj.dtm_tiles()
        build_region_dtm(self.ctx.name, self.ctx.aoi_gpkg, self.tile_index, dtm_tiles_dir)
        
    def build_terrain(self):
        raster_dir = self.ctx.rasters_dir()
        dtm_tiles_dir = self.proj.dtm_tiles()
        #processed_dtm(self.ctx.dtm_tile_path,raster_dir, self.ctx.name)
        build_terrain_for_region(raster_dir, dtm_tiles_dir,self.ctx.name)
        build_aspect_curve(raster_dir, self.ctx.name)

    def run(self):
        print(f"=== Building AOI for region {self.ctx.name} ===")
        self.build_region_aoi()
        print(f"=== Fetching houses for region {self.ctx.name} ===")
        self.fetch_houses()
        print(f"=== Fetching hydro and hazard data for region {self.ctx.name} ===")
        self.fetch_hydro_and_hazard()
        print(f"=== Clipping DTM to region {self.ctx.name} ===")
        self.clip_dtm_to_region()
        print(f"=== Building terrain derivatives for region {self.ctx.name} ===")
        self.build_terrain()
    
    def run_hyd_links(self):
        catchment_layer = "catchments"
        river_layer = "rivers"
        stations_layer = "vannforing_stasjoner"
        lakes_layer = "lakes"
        hydro_link_path = self.ctx.hydro_links_path()
        houses_layer = f"{self.ctx.name}_houses"
        hydro_links.build_house_hydro_links(self.ctx.houses_gpkg, houses_layer,self.ctx.hydro_gpkg,
                                            catchment_layer, river_layer, lakes_layer, stations_layer,
                                                    hydro_link_path)
    
    def ensure_hazard_links(self):
        links_dir = self.ctx.link_dir()
        flood_links_path = self.ctx.flood_links_path()
        landslide_links_path = self.ctx.landslide_links_path()
        links_dir.mkdir(parents=True, exist_ok=True)
    def run_links(self):
            houses = gpd.read_file(self.ctx.houses_gpkg, layer=f"{self.ctx.name}_houses")
            self.ensure_hazard_links()
            flood_link_path = self.ctx.flood_links_path()
            flood_labels.flood_links(houses, self.ctx.hazard_gpkg, flood_link_path)
            self.run_hyd_links()
            landslide_links.landslide_links(houses, self.ctx.hazard_gpkg, self.ctx.landslide_links_path())
            