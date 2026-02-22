import torch
from pathlib import Path
from src.utils.paths import ProjectPaths,RegionContext, GlobalContext, RasterPaths


import numpy as np
import pandas as pd

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

def aggregate_per_house(df, house_id_col="bygningsnummer"):
    if df is None or df.empty:
        return df
    if house_id_col not in df.columns:
        return df

    num_cols = [c for c in df.columns if c != house_id_col and pd.api.types.is_numeric_dtype(df[c])]
    bool_cols = [c for c in df.columns if c != house_id_col and df[c].dtype == "bool"]
    other_cols = [c for c in df.columns if c not in ([house_id_col] + num_cols + bool_cols)]

    agg = {}
    for c in num_cols:
        agg[c] = "median"
    for c in bool_cols:
        agg[c] = "max"     # any True -> True
    for c in other_cols:
        agg[c] = "first"   # keep first string/id-ish value

    out = df.groupby(house_id_col, as_index=False).agg(agg)
    return out

def keep_only(df: pd.DataFrame, keep: list[str], *, always_keep: str | None = None) -> pd.DataFrame:
    keep_set = set(keep)
    if always_keep:
        keep_set.add(always_keep)

    cols = [c for c in df.columns if c in keep_set]
    return df.loc[:, cols].copy()


def encode_raw_landslides(landslides_events: pd.DataFrame, house_df: pd.DataFrame,
                          house_id_col="bygningsnummer") -> pd.DataFrame:
    """
    0 = no events
    1 = one event
    2 = two+ events
    """
    if landslides_events is None or landslides_events.empty:
        out = house_df[[house_id_col]].copy()
        out["landslide_hazard_level"] = 0
        return out

    counts = landslides_events[house_id_col].value_counts()
    hazard = counts.apply(lambda x: 2 if x > 1 else 1)

    hazard_df = hazard.reset_index()
    hazard_df.columns = [house_id_col, "landslide_hazard_level"]

    merged = house_df[[house_id_col]].merge(hazard_df, on=house_id_col, how="left")
    merged["landslide_hazard_level"] = merged["landslide_hazard_level"].fillna(0).astype(int)
    return merged


def clean_link_tables(
    flood_links: pd.DataFrame | None = None,
    landslide_closest: pd.DataFrame | None = None,
    landslide_events_raw: pd.DataFrame | None = None,
    hydro_links: pd.DataFrame | None = None,
    *,
    house_id_col: str = "bygningsnummer",
) -> dict[str, pd.DataFrame]:

    out = {}

    if flood_links is not None:
        out["flood"] = keep_only(flood_links, FLOOD_KEEP, always_keep=house_id_col)

    if landslide_closest is not None:
        out["landslide_closest"] = keep_only(landslide_closest, LANDSLIDE_CLOSEST_KEEP, always_keep=house_id_col)

    if landslide_events_raw is not None:
        out["landslide_events_raw"] = keep_only(landslide_events_raw, LANDSLIDE_EVENT_RAW_KEEP, always_keep=house_id_col)

    if hydro_links is not None:
        out["hydro"] = keep_only(hydro_links, HYDRO_KEEP, always_keep=house_id_col)

    return out

import geopandas as gpd
from pathlib import Path

def load_houses_data(
    region: str,
    base_dir: str | Path = "master",
    target_epsg: int = 25833,
    house_id_col: str = "bygningsnummer",
    use_landslide_closest: bool = True,
):
    base_dir = Path(base_dir)

    houses_path = base_dir / f"raw/vector/houses/houses_{region}.gpkg"
    houses_layer = f"houses_{region}"

    houses = gpd.read_file(houses_path, layer=houses_layer).to_crs(epsg=target_epsg)
    houses = houses.set_geometry("geometry")
    houses[house_id_col] = houses[house_id_col].astype("int64")

    # link tables
    f_links = pd.read_parquet(base_dir / f"processed/links/flood_links_{region}.parquet")
    h_links = pd.read_parquet(base_dir / f"processed/links/hydro_links_{region}.parquet")

    # closest landslide table is optional
    l_closest = None
    if use_landslide_closest:
        l_closest_path = base_dir / f"processed/links/landslide_links_closest_{region}.parquet"
        if l_closest_path.exists():
            l_closest = pd.read_parquet(l_closest_path)

    # raw landslide events -> used for encoding target
    l_events = pd.read_parquet(base_dir / f"processed/links/landslide_links_{region}.parquet")

    cleaned = clean_link_tables(
        flood_links=f_links,
        landslide_closest=l_closest,
        landslide_events_raw=l_events,
        hydro_links=h_links,
        house_id_col=house_id_col,
    )

    f_clean = aggregate_per_house(cleaned["flood"], house_id_col)
    h_clean = aggregate_per_house(cleaned["hydro"], house_id_col)
    l_closest_clean = aggregate_per_house(cleaned.get("landslide_closest"), house_id_col)

    l_events_clean = cleaned.get("landslide_events_raw")
    l_encoded = encode_raw_landslides(l_events_clean, houses, house_id_col=house_id_col)
    l_encoded = aggregate_per_house(l_encoded, house_id_col)

    # merge links safely
    link_tables = [f_clean, h_clean, l_encoded]
    if l_closest_clean is not None:
        link_tables.append(l_closest_clean)

    links = link_tables[0]
    for nxt in link_tables[1:]:
        links = links.merge(nxt, on=house_id_col, how="left")

    houses_gdf = houses.merge(links, on=house_id_col, how="left")
    houses_gdf = gpd.GeoDataFrame(houses_gdf, geometry="geometry", crs=houses.crs)

    return houses_gdf

def load_graph_bundle(path):
    return torch.load(Path(path), map_location="cpu")
