import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

R_SMALL = 10
R_LARGE = 30

def window_stats(arr, row, col, radius):
    """Return basic stats for a square window around (row, col)."""
    nrows, ncols = arr.shape
    r0 = max(row - radius, 0)
    r1 = min(row + radius + 1, nrows)
    c0 = max(col - radius, 0)
    c1 = min(col + radius + 1, ncols)
    sub = arr[r0:r1, c0:c1]
    vals = sub[np.isfinite(sub)]
    if vals.size == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p90": float(np.percentile(vals, 90)),
    }


def sample_raster(src, x, y, r_small=R_SMALL, r_large=R_LARGE):
    """Sample value at (x, y) and stats in small/large windows."""
    arr = src.read(1).astype(float)
    # mask nodata to NaN
    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    row, col = src.index(x, y)

    # value at house
    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
        v_house = float(arr[row, col])
    else:
        v_house = np.nan

    stats_small = window_stats(arr, row, col, r_small)
    stats_large = window_stats(arr, row, col, r_large)

    return v_house, stats_small, stats_large

def build_features(region,houses, dtm_path, slope_path, twi_path, flow_path):
    # Try to find the ID column
    id_col = "bygningsnummer" if "bygningsnummer" in houses.columns else "bygningId"
    houses[id_col] = houses[id_col].astype(str)

    rows = []
    with rasterio.open(dtm_path) as dtm_src, \
         rasterio.open(slope_path) as slope_src, \
             rasterio.open(twi_path) as twi_src, \
                 rasterio.open(flow_path) as flow_src:

        for idx, row in houses.iterrows():
            hid = str(row[id_col])
            x = row.geometry.x
            y = row.geometry.y

            # file paths for this house
        
            rec = {"house_id": hid}

            # --- DTM1 elevation ---
            if dtm_path.exists():
                z_house, z_small, z_large = sample_raster(dtm_src, x, y)
                rec["z_house"] = z_house
                if z_small:
                    rec["z_mean_10m"] = z_small["mean"]
                    rec["z_std_10m"] = z_small["std"]
                if z_large:
                    rec["z_mean_30m"] = z_large["mean"]
                    rec["z_std_30m"] = z_large["std"]
                    rec["z_range_30m"] = z_large["max"] - z_large["min"]

                    # share above/below house in large window
                    arr = dtm_src.read(1).astype(float)
                    if dtm_src.nodata is not None:
                        arr[arr == dtm_src.nodata] = np.nan
                    r, c = dtm_src.index(x, y)
                    r0 = max(r - R_LARGE, 0)
                    r1 = min(r + R_LARGE + 1, arr.shape[0])
                    c0 = max(c - R_LARGE, 0)
                    c1 = min(c + R_LARGE + 1, arr.shape[1])
                    patch = arr[r0:r1, c0:c1]
                    vals = patch[np.isfinite(patch)]
                    if vals.size > 0 and np.isfinite(z_house):
                        rec["share_above_30m"] = float(np.sum(vals > z_house) / vals.size)
                        rec["share_below_30m"] = float(np.sum(vals < z_house) / vals.size)

            # --- slope ---
            if slope_path.exists():
                slope_house, s_small, s_large = sample_raster(slope_src, x, y)
                rec["slope_house"] = slope_house
                if s_small:
                    rec["slope_mean_10m"] = s_small["mean"]
                if s_large:
                    rec["slope_p90_30m"] = s_large["p90"]

            # --- TWI ---
            if twi_path.exists():
                twi_house, t_small, t_large = sample_raster(twi_src, x, y)
                rec["twi_house"] = twi_house
                if t_large:
                    rec["twi_p90_30m"] = t_large["p90"]

            # --- flow accumulation ---
            if flow_path.exists():
                fa_house, fa_small, fa_large = sample_raster(flow_src, x, y)
                rec["flowacc_house"] = fa_house
                if fa_large:
                    rec["flowacc_p90_30m"] = fa_large["p90"]

            rows.append(rec)
            if idx % 100 == 0:
                print(f"Processed {idx + 1}/{len(houses)} houses...")
        # close raster files

    df = pd.DataFrame(rows)
    
    return df

def merged_features(static_terrain, ml_cols, landslide_links, out_path,id_col="house_id"):
    # Merge on house_id
    df_landslides = pd.read_parquet(landslide_links)
    df_static_terrain = pd.read_parquet(static_terrain)
    merged = df_static_terrain.merge(df_landslides, on=id_col, how="left")
    ml_df = merged[[id_col] + ml_cols + ["num_landslides_100m"]]
    ml_df.to_parquet(out_path, index=False)