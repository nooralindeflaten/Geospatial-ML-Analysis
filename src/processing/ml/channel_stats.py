import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from pathlib import Path

def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str):
    if gdf.crs is None or gdf.crs.to_string() != target_crs:
        return gdf.to_crs(target_crs)
    return gdf

def compute_channel_stats(
    houses_gdf: gpd.GeoDataFrame,
    raster_paths,
    patch_size_pixels: int,
    target_crs: str = "EPSG:25833",
    sample_n: int = 2000,
    seed: int = 42,
    out_path = None,
):
    houses_gdf = ensure_crs(houses_gdf, target_crs)

    if len(houses_gdf) == 0:
        raise ValueError("houses_gdf is empty")

    rng = np.random.default_rng(seed)
    n = min(sample_n, len(houses_gdf))
    sample_idx = rng.choice(len(houses_gdf), size=n, replace=False)

    raster_keys = list(raster_paths.keys())
    rasters = {k: rasterio.open(str(p)) for k, p in raster_paths.items()}

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