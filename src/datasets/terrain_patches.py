import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import torch
from src.utils.paths import ProjectPaths, GlobalContext, RegionContext, RasterPaths
from src.utils.houses import Houses
from src.processing.terrain.patch_id_builder import build_patch_index
import pandas as pd

#master/scripts/test_model_note.ipynb
FLOOD_KEEP = [
    "region",
    "in_flom_analyseomraade",
    "hazard_class_flom",
]

LANDSLIDE_CLOSEST_KEEP = [
    "inside_source_area",
    "inside_runout_area",
    "any_landslide_area_inside",

    "dist_to_trigger_point",
    "dist_to_runout_point",
    "dist_to_source_area",
    "dist_to_runout_area",
    "dist_to_landslide_event",

    "sourcearea_area_m2",
    "runoutarea_area_m2",

    "trigger_point_far",
    "runout_point_far",
    "landslide_event_far",
]

LANDSLIDE_EVENT_RAW_KEEP = [
    "skredID",
    "distance_m",
    "skredType",
    "skredTidspunkt",
]

HYDRO_KEEP = [
    "dist_to_river",
    "dist_to_lake",
    "dist_to_hyd",
    "arealEnhet_km2",
    "arealTotal_km2",
    "QNormal_lskm2",
    "QNormal_Mm3Aar",
    "QNormalOppstrm_Mm3Aar",
    "elveordenstrahler",
    "arealregineenhet_km2",
    "areal_km2",
    "arealNorge_km2",
    "nedborfeltareal_km2",
    "minsteVannforing",
]
def extract_stack_patches(region,rasters, houses_gdf, patches_m):
        """ 
        open rasters and extract patches around houses
        """
        keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
        srcs = {k: rasterio.open(rasters[k]) for k in keys}
        
        dem_src = srcs["dtm"]
        res_x, res_y = dem_src.res
        assert abs(res_x - res_y) < 1e-6, "Rasters must have square pixels"
        res = res_x
        if houses_gdf.crs != dem_src.crs:
            houses_gdf = houses_gdf.to_crs(dem_src.crs)
        half_patch_size = int((patches_m / res) / 2)
        patch_pixels = half_patch_size * 2
        
        patches = []
        keep_rows = []
        
        for idx, row in tqdm(houses_gdf.iterrows(), total=len(houses_gdf), desc=f"{region} patches"):
            geometry = row.geometry
            x, y = geometry.x, geometry.y
            row_idx, col_idx = dem_src.index(x, y)
            
            window = Window(
                col_off=col_idx - half_patch_size,
                row_off=row_idx - half_patch_size,
                width=patch_pixels,
                height=patch_pixels
            )
            
            chans = []
            ok = True
            for k in keys:
                arr = srcs[k].read(1, window=window)
                if arr.shape != (patch_pixels, patch_pixels):
                    ok = False
                    break
                chans.append(arr)

            if not ok:
                continue

            stack = np.stack(chans, axis=0).astype("float32")
            
            stack = np.where(np.isfinite(stack), stack, np.nan)
            for c in range(stack.shape[0]):
                med = np.nanmedian(stack[c])
                stack[c] = np.where(np.isnan(stack[c]), med, stack[c])

            patches.append(stack)
            keep_rows.append(idx)

        # close rasters
        for s in srcs.values():
            s.close()

        houses_ok = houses_gdf.loc[keep_rows].copy()

        if len(patches) == 0:
            patches_t = torch.empty((0, len(keys), patch_pixels, patch_pixels), dtype=torch.float32)
        else:
            patches_t = torch.from_numpy(np.stack(patches, axis=0)).float()

        return houses_ok, patches_t


def extract_stack_for_xy(
    x: float,
    y: float,
    raster_sources,
    patch_size_m: float,
):
    dtm_src = raster_sources[0]

    res = float(dtm_src.res[0])
    half_pixels = int((patch_size_m / 2) / res)
    patch_pixels = half_pixels * 2

    row_idx, col_idx = dtm_src.index(x, y)

    window = Window(
        col_idx - half_pixels,
        row_idx - half_pixels,
        patch_pixels,
        patch_pixels,
    )

    patches = [src.read(1, window=window) for src in raster_sources]

    if any(p.shape != (patch_pixels, patch_pixels) for p in patches):
        return None, patch_pixels

    stack = np.stack(patches, axis=0).astype("float32")

    stack = np.where(np.isfinite(stack), stack, np.nan)
    for c in range(stack.shape[0]):
        chan = stack[c]
        med = np.nanmedian(chan)
        if np.isnan(med):
            med = 0.0
        stack[c] = np.where(np.isnan(chan), med, chan)

    return stack, patch_pixels

class TerrainPatches:
    def __init__(self, split, houses_ctx: Houses,raster_ctx: RasterPaths,region_ctx: RegionContext, label_col,random_state=42):
        self.houses_ctx = houses_ctx
        self.region = region_ctx.name
        self.idx_parquet = region_ctx.patch_index_parquet
        self.rasters = raster_ctx
        self.houses_gdf = houses_ctx.houses_gdf
        self.patch_size_m = houses_ctx.patch_size_m()
        self.tabular_cols = HYDRO_KEEP + FLOOD_KEEP + LANDSLIDE_CLOSEST_KEEP
        self.id_col = houses_ctx.id_col
        self.label_col = label_col
        self.normalize = True
        if self.idx_parquet.exists():
            self.df = pd.read_parquet(self.idx_parquet)
        else:
            self.df = build_patch_index(
                houses_gpkg=region_ctx.houses_gpkg,
                houses_layer=region_ctx.house_layer,
                label_col=label_col,
                flood_p=region_ctx.flood_links_path(),
                id_col=self.id_col,
                target_crs="EPSG:25833",
                val_frac=0.2,
                stratify=True,
                random_state=random_state,
                out_index_parquet=region_ctx.patch_index_parquet(),
            )

        # Filter split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self._sources = None
    
    def __len__(self):
        return len(self.df)

    def _open_sources(self):
        if self._sources is None:
            paths = [str(p) for p in self.rasters.as_list()]
            self._sources = [rasterio.open(p) for p in paths]
        return self._sources

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = float(row["x"]), float(row["y"])
        label = int(row[self.label_col])

        sources = self._open_sources()
        stack, patch_pixels = extract_stack_for_xy(
            x=x, y=y,
            raster_sources=sources,
            patch_size_m=self.patch_size_m,
        )

        if stack is None:
            c = len(sources)
            stack = torch.zeros((c, patch_pixels, patch_pixels), dtype=torch.float32)
        else:
            stack = torch.from_numpy(stack)

        if self.normalize:
            # simple per-channel standardization
            # (safe + fast, avoids storing dataset stats)
            eps = 1e-6
            for c in range(stack.shape[0]):
                chan = stack[c]
                mean = chan.mean()
                std = chan.std()
                stack[c] = (chan - mean) / (std + eps)

        y_t = torch.tensor(label, dtype=torch.long)
        return stack, y_t

    def close(self):
        if self._sources is not None:
            for s in self._sources:
                s.close()
            self._sources = None