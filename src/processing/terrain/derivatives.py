import numpy as np
import rasterio
import richdem as rd
from pathlib import Path
import subprocess

def run(cmd):
    print("Running:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)
    
def processed_dtm(dmt_tile, dtm_processed, region, nodata=-9999, res=10.0, crs="EPSG:25833"):
    out = dtm_processed / f"{region}_dtm10_processed.tif"
    run([
            "gdalwarp",
            "-t_srs", crs,
            "-tr", str(res), str(res),
            "-tap",
            "-r", "bilinear",
            "-dstnodata", str(nodata),
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            str(dmt_tile),
            str(out),
        ])
    
def clean_dem(dem: rd.rdarray) -> rd.rdarray:
    """
    converts to float64
    finds the minimum finite elevation
    replaces any non-finite or no_data values with that min
    """
    arr = np.array(dem, dtype="float64")

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise RuntimeError("DEM has no finite values")

    vmin = float(arr[finite_mask].min())

    bad = ~finite_mask
    if bad.any():
        print(f"  [clean_dem] Found {bad.sum()} non-finite cells; replacing with {vmin}")
        arr[bad] = vmin

    dem_clean = rd.rdarray(arr, no_data=vmin)
    dem_clean.geotransform = dem.geotransform
    dem_clean.projection = dem.projection
    return dem_clean


def build_terrain_for_region(dtm_tiles,dtm_dir,region: str):
    dtm_tiles.mkdir(parents=True, exist_ok=True)

    dtm10 = dtm_dir / f"{region}_dtm10.tif"
    if not dtm10.exists():
        raise FileNotFoundError(f"Missing raw DTM for {region}: {dtm10}")

    out_filled   = dtm_tiles / f"{region}_dtm10_filled.tif"
    out_slope    = dtm_tiles / f"{region}_slope_deg.tif"
    out_flowacc  = dtm_tiles / f"{region}_flowacc_d8.tif"
    out_twi      = dtm_tiles / f"{region}_twi.tif"
    
    print(f"\n=== Processing region: {region} ===")
    print(f"Loading DEM from {dtm10} ...")
    dem_raw = rd.LoadGDAL(str(dtm10))

    print("Cleaning DEM (fix NaNs / nodata) ...")
    dem = clean_dem(dem_raw)
    print("Filling depressions (hydro-correct DEM) ...")
    dem_filled = rd.FillDepressions(dem, epsilon=False, in_place=False)

    print(f"Saving filled DEM to {out_filled} ...")
    rd.SaveGDAL(str(out_filled), dem_filled)

    print(f"Saving filled DEM to {out_filled} ...")
    rd.SaveGDAL(str(out_filled), dem_filled)


    print("Computing slope (degrees) ...")
    slope_deg = rd.TerrainAttribute(dem_filled, attrib="slope_degrees")

    print(f"Saving slope raster to {out_slope} ...")
    rd.SaveGDAL(str(out_slope), slope_deg)

   
    print("Computing flow accumulation (D8) ...")
    flowacc = rd.FlowAccumulation(dem_filled, method="D8")

    print(f"Saving flow accumulation raster to {out_flowacc} ...")
    rd.SaveGDAL(str(out_flowacc), flowacc)

    print("Computing TWI ...")

    slope_rad = rd.TerrainAttribute(dem_filled, attrib="slope_radians")

    gt = dem_filled.geotransform
    cellsize = float(gt[1])

    fa = np.array(flowacc, dtype="float64")
    sr = np.array(slope_rad, dtype="float64")

    eps = 1e-6

    As = (fa + 1.0) * cellsize

    # TWI formula
    twi_vals = np.log((As + eps) / (np.tan(sr) + eps))

    twi_vals = np.where(np.isfinite(twi_vals), twi_vals, np.nan)
    twi = rd.rdarray(twi_vals, no_data=np.nan)
    twi.geotransform = dem_filled.geotransform
    twi.projection = dem_filled.projection

    print(f"Saving TWI raster to {out_twi} ...")
    rd.SaveGDAL(str(out_twi), twi)
    
    print("Done!")

def write_raster(template_path: Path, arr: np.ndarray, out_path: Path):
    with rasterio.open(template_path) as src:
        profile = src.profile

    profile.update(
        dtype="float64",
        nodata=np.nan,
        count=1,
        compress="LZW",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("float64"), 1)

    print("Wrote", out_path)

def clean(arr):
    """
    Replace non-finite with NaN, then fill NaNs with median.
    Keeps output stable for ML and visualization.
    """
    arr = np.array(arr, dtype="float64")
    arr = np.where(np.isfinite(arr), arr, np.nan)

    med = np.nanmedian(arr)
    if np.isnan(med):
        # worst-case fallback
        med = 0.0

    arr = np.where(np.isnan(arr), med, arr)
    return arr.astype("float64")
    
def build_aspect_curve(dtm_tiles,region: str):

    dtm_filled = dtm_tiles / f"{region}_dtm10_filled.tif"
    if not dtm_filled.exists():
        raise FileNotFoundError(f"Missing raw DTM for {region}: {dtm_filled}")

    out_aspect = dtm_tiles / f"{region}_aspect_deg.tif"
    out_curv = dtm_tiles / f"{region}_curvature.tif"
    print(f"\n=== {region.upper()} ===")
    print("Loading filled DEM from", dtm_filled)

    with rasterio.open(dtm_filled) as src:
        dem = src.read(1).astype("float64")
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # richdem object
    rd_dem = rd.rdarray(dem, no_data=np.nan)

    print("Computing aspect (degrees)...")
    aspect = rd.TerrainAttribute(rd_dem, attrib="aspect")  # 0–360

    print("Computing curvature...")
    curvature = rd.TerrainAttribute(rd_dem, attrib="curvature")

    aspect = clean(aspect)
    curvature = clean(curvature)

    write_raster(dtm_filled, aspect, out_aspect)
    write_raster(dtm_filled, curvature, out_curv)

    print("Done.")

