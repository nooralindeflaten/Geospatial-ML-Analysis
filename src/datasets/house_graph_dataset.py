from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torch_geometric.transforms import RandomNodeSplit
import torch_geometric
import torch
import rasterio
from rasterio.windows import Window
import numpy as np
import geopandas as gpd
from src.utils.houses import Houses
from src.utils.paths import RasterPaths, RegionContext, GlobalContext, ProjectPaths
from src.processing.ml.assemble_graph import build_data_objects
class TerrainBuilderStack:
    def __init__(self, region, raster_paths, houses_gdf,patches_m=200):
        self.region = region
        self.raster_paths = raster_paths  # dict: {"dtm": Path, ...}
        self.houses_gdf = houses_gdf
        self.patches_m = patches_m
        self._sources = None
        self.label_flood = "hazard_class_flom"
        self.label_landslide = "landslide_hazard_level"

    def _open_sources(self):
        if self._sources is None:
            keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
            self._sources = {k: rasterio.open(self.raster_paths[k]) for k in keys}

            # ensure house CRS matches rasters
            dtm_crs = self._sources["dtm"].crs
            if self.houses_gdf.crs != dtm_crs:
                self.houses_gdf = self.houses_gdf.to_crs(dtm_crs)

        return self._sources

    def close(self):
        if self._sources:
            for src in self._sources.values():
                src.close()
        self._sources = None

    def extract_one_patch(self, geom):
        srcs = self._open_sources()
        dtm_src = srcs["dtm"]

        res_x, res_y = dtm_src.res
        assert abs(res_x - res_y) < 1e-6, "Rasters must have square pixels"
        res = float(res_x)

        half_pixels = int((self.patches_m / 2) / res)
        patch_pixels = half_pixels * 2

        row_idx, col_idx = dtm_src.index(geom.x, geom.y)

        window = Window(
            col_idx - half_pixels,
            row_idx - half_pixels,
            patch_pixels,
            patch_pixels,
        )

        keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
        patches = [srcs[k].read(1, window=window) for k in keys]

        # Skip edge cases
        if any(p.shape != (patch_pixels, patch_pixels) for p in patches):
            return None

        stack = np.stack(patches, axis=0).astype("float32")

        stack = np.where(np.isfinite(stack), stack, np.nan)
        for c in range(stack.shape[0]):
            med = np.nanmedian(stack[c])
            if np.isnan(med):
                med = 0.0
            stack[c] = np.where(np.isnan(stack[c]), med, stack[c])

        return stack
    
    

class HouseGraphDataset(Dataset):
    def __init__(
        self,
        region,
        raster_paths,
        base_dir="../",
        k=8,
        flood_label="hazard_class_flom",
        landslide_label="landslide_hazard_level",
        graph_path="",
        patches_m=100,
    ):
        self.region = region
        self.base_dir = base_dir
        self.k = k
        self.flood_label = flood_label
        self.landslide_label = landslide_label
        self.graph_path = graph_path
        self.patches_m = patches_m
        self.raster_paths = raster_paths

        data, houses_ok, feature_cols, graph = build_data_objects(
            region,
            house_id_col="bygningsnummer",
            base_dir=base_dir,
            flood_label=flood_label,
            landslide_label=landslide_label,
            graph_path=graph_path,
        )

        self.data = data
        self.houses_ok = houses_ok
        self.feature_cols = feature_cols
        self.graph = graph

        self.terrain = TerrainBuilderStack(
            region=region,
            raster_paths=self.raster_paths,
            houses_gdf=self.houses_ok,
            patches_m=patches_m,
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data

    def get_patches_for_node_idx(self, node_idx_tensor):
        patches = []
        for i in node_idx_tensor.tolist():
            geom = self.houses_ok.iloc[int(i)].geometry
            patch = self.terrain.extract_one_patch(geom)

            if patch is None:
                # fallback zeros
                # infer size from dtm resolution
                dtm_src = self.terrain._open_sources()["dtm"]
                res = float(dtm_src.res[0])
                half = int((self.patches_m / 2) / res)
                px = half * 2
                patch = np.zeros((6, px, px), dtype="float32")

            patches.append(torch.from_numpy(patch).float())

        return torch.stack(patches, dim=0)

    def close(self):
        self.terrain.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass