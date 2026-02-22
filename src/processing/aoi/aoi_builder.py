from pathlib import Path
import geopandas as gpd
def build_region_from_kommuner(region_name, kommune_names,kommune_aoi,out_path):
    """
    Build a single region polygon by sectioning and dissolving multiple kommuner.
    """

    komm = gpd.read_file(kommune_aoi,layer='app:Kommune').to_crs(25833)

    name_cols = [c for c in komm.columns if "kommunenavn" in c.lower()]
    if not name_cols:
        raise RuntimeError(f"Could not find a kommune name column in {komm.columns}")
    name_col = name_cols[0]

    subset = komm[komm[name_col].isin(kommune_names)].copy()
    if subset.empty:
        raise RuntimeError(f"No kommuner matched {kommune_names} in column {name_col}")

    subset = subset.to_crs("EPSG:25833")

    dissolved = subset.dissolve(by=None)
    region_geom = dissolved.geometry.iloc[0]

    region_gdf = gpd.GeoDataFrame(
        {"region": [region_name]},
        geometry=[region_geom],
        crs="EPSG:25833",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region_gdf.to_file(out_path, layer="region", driver="GPKG")
    print(f"Saved region '{region_name}' to {out_path}")
