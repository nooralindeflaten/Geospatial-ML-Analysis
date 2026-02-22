from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ANALYSE_LAYER = "flomsone_analyseomrade"

HIGH_LAYERS = [
    "flomsone_10",        # 10-year flood
    "flomsone_20",        # 20-year
    "flomsone_50",        # 50-year
    "flomsone_100",       # 100-year
    "flomsone_20_klima",  # 20-year with climate
]

MEDIUM_LAYERS = [
    "flomsone_200",        # 200-year
    "flomsone_500",        # 500-year
    "flomsone_1000",       # 1000-year
    "flomsone_200_klima",  # 200-year with climate
    "flomsone_1000_klima", # 1000-year with climate
]


def mark_intersections(houses: gpd.GeoDataFrame,
                       hazard_gpkg: Path,
                       layer_name: str,
                       class_value: int,
                       only_from: float | None = None) -> None:
    """
    Update houses['hazard_class_flom'] in-place for houses that intersect
    the given flomsone layer.

    Parameters
    ----------
    houses : GeoDataFrame
        Must have 'geometry', 'hazard_class_flom', and 'in_flom_analyseomraade'.
    layer_name : str
        Layer in hazard_gpkg to read.
    class_value : int
        Class to assign (1 for medium, 2 for high).

    """
    print(f"  - marking layer '{layer_name}' as class {class_value}")
    flom = gpd.read_file(hazard_gpkg, layer=layer_name)

    # Reproject to match houses
    if flom.crs is None or flom.crs.to_string() != houses.crs.to_string():
        flom = flom.to_crs(houses.crs)

    # Spatial join: which houses intersect this flomsone layer?
    joined = gpd.sjoin(
        houses[["geometry"]],
        flom[["geometry"]],
        how="left",
        predicate="intersects",
    )

    idx = joined.index[joined["index_right"].notna()].unique()

    if only_from is None:
        mask = houses.index.isin(idx) & houses["in_flom_analyseomraade"]
    else:
        mask = (
            houses.index.isin(idx)
            & houses["in_flom_analyseomraade"]
            & (houses["hazard_class_flom"] == only_from)
        )

    houses.loc[mask, "hazard_class_flom"] = class_value

def flood_links(houses, hazard_gpkg: Path, out_labels: Path, target_crs: str = "EPSG:25833"):
    # Ensure CRS
    if houses.crs is None or houses.crs.to_string() != target_crs:
        print(f"Reprojecting houses to {target_crs}...")
        houses = houses.to_crs(target_crs)

    # Load analyseområde
    print(f"Loading flomsone analyseområde from {hazard_gpkg} ({ANALYSE_LAYER})...")
    analyse = gpd.read_file(hazard_gpkg, layer=ANALYSE_LAYER)
    if analyse.crs is None or analyse.crs.to_string() != houses.crs.to_string():
        analyse = analyse.to_crs(houses.crs)

    # Mark houses inside analyseområde
    print("Marking houses inside FlomsoneAnalyseomraade...")
    joined = gpd.sjoin(
        houses[["geometry"]],
        analyse[["geometry"]],
        how="left",
        predicate="within",
    )
    in_area = joined["index_right"].notna().to_numpy()

    houses["in_flom_analyseomraade"] = in_area

    # Init hazard class:
    # - NaN     outside analyseområde (kept as NaN)
    # - 0       inside analyseområde, baseline (no flomsone yet)
    houses["hazard_class_flom"] = np.nan
    houses.loc[houses["in_flom_analyseomraade"], "hazard_class_flom"] = 0

    # 1) Medium zones (only set where currently 0)
    print("Assigning medium hazard class (1) from layers:", MEDIUM_LAYERS)
    for lname in MEDIUM_LAYERS:
        mark_intersections(houses, hazard_gpkg,lname, class_value=1, only_from=0)

    # 2) High zones (override whatever is there, as long as inside analyseomraade)
    print("Assigning high hazard class (2) from layers:", HIGH_LAYERS)
    for lname in HIGH_LAYERS:
        mark_intersections(houses, hazard_gpkg,lname, class_value=2, only_from=None)

    # Quick sanity check counts
    counts = houses["hazard_class_flom"].value_counts(dropna=False).sort_index()
    print("hazard_class_flom counts (including NaN):")
    print(counts)

    # Save outputs
    
    # Drop geometry for ML-friendly table
    df = pd.DataFrame(houses.drop(columns="geometry"))
    print(df.head())
    df.to_parquet(out_labels, index=False)

    print("Done.")