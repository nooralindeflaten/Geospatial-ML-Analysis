from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from tqdm import tqdm
PATCH_SIZE_M = 200.0

def build_house_stacks(gdf,patches_dir,dtm_path, slope_path, aspect_path, curv_path, fa_path, twi_path):
    gdf.set_geometry("geometry")
    #df = df[df["hazard_class_flom"].notna()].copy()
    gdf["bygningsnummer"] = gdf["bygningsnummer"].astype(int)

    with rasterio.open(dtm_path) as dem_src, \
         rasterio.open(slope_path) as slope_src, \
         rasterio.open(aspect_path) as aspect_src, \
         rasterio.open(curv_path) as curv_src, \
         rasterio.open(fa_path) as fa_src, \
         rasterio.open(twi_path) as twi_src:

        res_x, res_y = dem_src.res  # should be ~10,10 for DTM10
        assert abs(res_x - res_y) < 1e-6
        res = res_x

        half_pixels = int((PATCH_SIZE_M / 2) / res)  # e.g. 100/10 = 10
        patch_pixels = half_pixels * 2               # e.g. 20x20

        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting house stacks"):
            bid = int(row["bygningsnummer"])
            out_path = patches_dir / f"house_{bid}.npy"
            if out_path.exists():
                continue

            geom = row.geometry

            # Convert coord -> raster indices
            row_idx, col_idx = dem_src.index(geom.x, geom.y)

            window = Window(
                col_idx - half_pixels,
                row_idx - half_pixels,
                patch_pixels,
                patch_pixels,
            )

            dem_patch    = dem_src.read(1, window=window)
            slope_patch  = slope_src.read(1, window=window)
            aspect_patch = aspect_src.read(1, window=window)
            curv_patch   = curv_src.read(1, window=window)
            fa_patch     = fa_src.read(1, window=window)
            twi_patch    = twi_src.read(1, window=window)

            if dem_patch.shape != (patch_pixels, patch_pixels):
                # skip houses near the raster edge
                continue

            stack = np.stack(
                [dem_patch, slope_patch, aspect_patch, curv_patch, fa_patch, twi_patch],
                axis=0,
            ).astype("float32")

            # basic NaN handling
            stack = np.where(np.isfinite(stack), stack, np.nan)
            for c in range(stack.shape[0]):
                chan = stack[c]
                med = np.nanmedian(chan)
                stack[c] = np.where(np.isnan(chan), med, chan)

            stack
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, stack)

def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str):
    if gdf.crs is None or gdf.crs.to_string() != target_crs:
        return gdf.to_crs(target_crs)
    return gdf



class SavedPatchesDataset:
    def __init__(self, houses,raster_paths,patches_dir,labels_path,split: str = "train", normalize: bool = True):
        self.houses = houses
        self.labels_path = labels_path
        self.raster_paths = raster_paths
        self.dtm_path = raster_paths["dtm"]
        self.slope_path = raster_paths["slope"]
        self.aspect_path = raster_paths["aspect"]
        self.curv_path = raster_paths["curv"]
        self.flowacc_path = raster_paths["flowacc"]
        self.twi_path = raster_paths["twi"]
        
        if not patches_dir.exists():
            print(f"Patches directory {patches_dir} does not exist, building patches...")
            build_house_stacks(
                houses,
                patches_dir,
                self.dtm_path,
                self.slope_path,
                self.aspect_path,
                self.curv_path,
                self.flowacc_path,
                self.twi_path,
            )
            
    def compute_channel_stats(self,
        patch_size_pixels: int,
        target_crs: str = "EPSG:25833",
        sample_n: int = 2000,
        seed: int = 42,
    ):
        houses_gdf = ensure_crs(self.houses, target_crs)

        if len(houses_gdf) == 0:
            raise ValueError("houses_gdf is empty")

        rng = np.random.default_rng(seed)
        n = min(sample_n, len(houses_gdf))
        sample_idx = rng.choice(len(houses_gdf), size=n, replace=False)

        raster_keys = list(self.raster_paths.keys())
        rasters = {k: rasterio.open(str(p)) for k, p in self.raster_paths.items()}

        half = patch_size_pixels // 2
        c = len(raster_keys)

        # Welford stats per channel
        count = np.zeros(c, dtype=np.int64)
        mean = np.zeros(c, dtype=np.float64)
        m2 = np.zeros(c, dtype=np.float64)

        def window_for_point(src, x, y):
            row_idx, col_idx = src.index(x, y)
            return Window(col_idx - half, row_idx - half, patch_size_pixels, patch_size_pixels)

        try:
            for i in sample_idx:
                geom = houses_gdf.iloc[i].geometry

                for ci, k in enumerate(raster_keys):
                    src = rasters[k]
                    win = window_for_point(src, geom.x, geom.y)

                    arr = src.read(
                        1,
                        window=win,
                        boundless=True,
                        fill_value=np.nan,
                    ).astype("float32")

                    vals = arr[np.isfinite(arr)]
                    if vals.size == 0:
                        continue

                    # update channel stats with batch of values
                    for v in vals:
                        count[ci] += 1
                        delta = v - mean[ci]
                        mean[ci] += delta / count[ci]
                        delta2 = v - mean[ci]
                        m2[ci] += delta * delta2

        finally:
            for src in rasters.values():
                try:
                    src.close()
                except Exception:
                    pass

        # avoid divide-by-zero
        var = np.where(count > 1, m2 / (count - 1), 0.0)
        std = np.sqrt(var)

        stats = {
            "keys": np.array(raster_keys),
            "mean": mean.astype("float32"),
            "std": std.astype("float32"),
            "count": count,
            "patch_size_pixels": np.array([patch_size_pixels]),
            "sample_n": np.array([n]),
        }

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **stats)
            print("Wrote channel stats:", out_path)
            print("keys:", raster_keys)
            print("mean:", stats["mean"])
            print("std:", stats["std"])

        return stats        
        
class StaticPatchesDataset:
        def __init__(self, patches_dir,stats_path,labels_path,split: str = "train", normalize: bool = True):
            df = pd.read_parquet(labels_path)
            df = df[df["hazard_class_flom"].notna()].copy()
            df["hazard_class_flom"] = df["hazard_class_flom"].astype(int)

            # if you have a split column, use it; else for now just use all
            if "split" in df.columns:
                df = df[df["split"] == split]

            df["feature_path"] = df["bygningsnummer"].astype(int).apply(
                lambda bid: patches_dir / f"house_{bid}.npy"
            )

            mask = df["feature_path"].apply(lambda p: p.exists())
            missing = (~mask).sum()
            if missing > 0:
                print(f"[WARN] {missing} feature stacks missing, dropping those rows")
            df = df[mask].reset_index(drop=True)

            self.df = df
            self.normalize = normalize
            self.stats_path = stats_path
            self.patches_dir = patches_dir
            stats = np.load(stats_path)
            self.channel_mean = stats["mean"]  # (C,)
            self.channel_std  = stats["std"]   # (C,)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            bid = int(row["bygningsnummer"])
            arr = np.load(self.patches_dir / f"house_{bid}.npy")  # (C,H,W)
            if self.normalize:
                mean = self.channel_mean[:, None, None]
                std = self.channel_std[:, None, None]
                arr = (arr - mean) / (std + 1e-6)
            x = torch.from_numpy(arr).float()
            y = torch.tensor(row["hazard_class_flom"], dtype=torch.long)
            return x, y