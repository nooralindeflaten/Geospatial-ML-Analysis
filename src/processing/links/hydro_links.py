from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


CATCH_KEEP_COLS = [
    "objectid", "regine_no", "regine", "vassdragsnr",
    "felt_id", "feltkode", "feltareal", "arealenhet_km2",
]
RIVER_KEEP_COLS = [
    "objectid", "vassdragsnr", "elvenavn", "strekninglnr",
    "elvenavnhierarki", "elv_lengde", "elv_fall",
]
LAKE_KEEP_COLS = [
    "objectid", "vatnlnr", "navn", "areal_m2", "omkrets_m",
]
STATION_KEEP_COLS = [
    "stationId", "stationName", "regineNo", "drainageBasinArea",
    "drainageBasinKey", "utmEast_Z33", "utmNorth_Z33",
]

HYD_KEEP_COLS = [
    "OBJECTID",
    "stasjonNr",
    "stasjonNavn",
    "stasjonType",
    "sanntid",
    "stasjonStatus",
    "vassdragsNr",
    "vassdragsOmradeNr",
    "vassdragsomrade",
    "elvenavnHierarki",
    "totalt_feltareal_km2",
    "elvetetthet",
    "gradient_10_85",
    "feltgradient",
    "elvegradient",
    "normal_arsavrenning_91_20",
    "andel_jordbruksareal",
    "andel_myrareal",
    "andel_skogareal",
    "andel_breareal",
    "andel_innsjoareal",
    "andel_snaufjell",
    "andel_urbantareal",
]

def _subset_cols(df, keep_cols):
    """Return df with only columns that exist in df."""
    cols = [c for c in keep_cols if c in df.columns]
    # Always keep geometry if present
    if "geometry" in df.columns and "geometry" not in cols:
        cols.append("geometry")
    return df[cols]


def load_houses(houses_gpkg,houses_layer):
    gdf = gpd.read_file(houses_gpkg, layer=houses_layer)
    # Normalize ID column
    if "house_id" in gdf.columns:
        gdf["bygningsnummer"] = gdf["house_id"].astype(str)
    elif "bygningId" in gdf.columns:
        gdf = gdf.rename(columns={"bygningId": "house_id"})
        gdf["house_id"] = gdf["house_id"].astype(str)
    elif "id" in gdf.columns:
        gdf = gdf.rename(columns={"id": "house_id"})
        gdf["house_id"] = gdf["house_id"].astype(str)
    else:
        raise ValueError("No obvious house ID column found (tried house_id, bygningId, id).")

    return gdf


def load_hydro_vectors(houses_crs,hydro_gpkg,catchment_layer,rivers_layer,lakes_layer,hyd_layer):
    # Catchments
    catch = gpd.read_file(hydro_gpkg, layer=catchment_layer)
    if catch.crs != houses_crs:
        catch = catch.to_crs(houses_crs)
    catch["catchment_id"] = catch["vassdragsNr"].astype(str)
    catch = _subset_cols(catch, CATCH_KEEP_COLS)

    # Rivers
    rivers = gpd.read_file(hydro_gpkg, layer=rivers_layer)
    if rivers.crs != houses_crs:
        rivers = rivers.to_crs(houses_crs)
    rivers["riverId"] = rivers["objectid"]
    rivers = _subset_cols(rivers, RIVER_KEEP_COLS)

    # Lakes
    lakes = gpd.read_file(hydro_gpkg, layer=lakes_layer)
    if lakes.crs != houses_crs:
        lakes = lakes.to_crs(houses_crs)
    lakes = _subset_cols(lakes, LAKE_KEEP_COLS)

    # Hydro stations
    hydro_station = gpd.read_file(hydro_gpkg, layer=hyd_layer)
    if hydro_station.crs != houses_crs:
        hydro_station = hydro_station.to_crs(houses_crs)
    hydro_station = _subset_cols(hydro_station, HYD_KEEP_COLS)

    return catch, rivers, lakes, hydro_station

def build_house_hydro_links(houses_gpkg,houses_layer,hydro_gpkg,catchment_layer,rivers_layer,lakes_layer,hyd_layer,out_parquet):
    # 1) Load houses
    houses = load_houses(houses_gpkg,houses_layer)
    houses = houses.set_geometry("geometry")
    houses_crs = houses.crs
    print(f"Houses: {len(houses)} features, CRS = {houses_crs}")

    # 2) Load hydro
    catch, rivers, lakes, hydro_station = load_hydro_vectors(houses_crs,hydro_gpkg,catchment_layer,rivers_layer,lakes_layer,hyd_layer)
    print(f"Catchments: {len(catch)}, Rivers: {len(rivers)}, Lakes: {len(lakes)}, Hydro stations: {len(hydro_station)}")

    # ----------------------
    # House -> Catchment
    # ----------------------
    if not catch.empty:
        houses_catch = gpd.sjoin(
            houses,
            catch,
            how="left",
            predicate="within",
        )
        # rename joined columns to avoid clashes
        houses_catch = houses_catch.drop(columns=["index_right"])
        # we keep the original house geom; catchment geom not needed here
        # but we already subset catch so it's ok.
    else:
        houses_catch = houses.copy()

    # ----------------------
    # House -> nearest River
    # ----------------------
    if not rivers.empty:
        rivers_no_geom = rivers.drop(columns=[c for c in ["geometry"] if c in rivers.columns])
        rivers_for_join = rivers[["geometry"]].join(rivers_no_geom)

        house_riv = gpd.sjoin_nearest(
            houses_catch,
            rivers_for_join,
            how="left",
            distance_col="dist_to_river",
        )
    else:
        house_riv = houses_catch.copy()
        house_riv["dist_to_river"] = pd.NA
        # drop helper index from this join so it doesn't clash later
    if "index_right" in house_riv.columns:
        house_riv = house_riv.drop(columns=["index_right"])


    # To avoid geometry duplication when exporting to Parquet, we will drop geometry at the very end.
    if not hydro_station.empty:
        hyd_no_geom = hydro_station.drop(columns=[c for c in ["geometry"] if c in hydro_station.columns])
        hyd_for_join = hydro_station[["geometry"]].join(hyd_no_geom)

        house_hyd = gpd.sjoin_nearest(
            house_riv,
            hyd_for_join,
            how="left",
            distance_col="dist_to_hyd",
        )
    else:
        house_hyd = house_riv.copy()
        house_hyd['dist_to_hyd'] = pd.NA
        
    if "index_right" in house_hyd.columns:
        house_hyd = house_hyd.drop(columns=["index_right"])

#----------------------
    # House -> nearest Station
    # --------------------------------------
    # Clean up and export
    # ----------------------
    # Drop any extra spatial join helper columns
        # ----------------------
    # House -> nearest Lake
    # ----------------------
    if not lakes.empty:
        lakes_no_geom = lakes.drop(columns=[c for c in ["geometry"] if c in lakes.columns])
        lakes_for_join = lakes[["geometry"]].join(lakes_no_geom)

        house_lake = gpd.sjoin_nearest(
            house_hyd,
            lakes_for_join,
            how="left",
            distance_col="dist_to_lake",
        )
    else:
        house_lake = house_hyd.copy()
        house_lake["dist_to_lake"] = pd.NA

    if "index_right" in house_lake.columns:
        house_lake = house_lake.drop(columns=["index_right"])

    # Final frame after rivers, hydro stations, and lakes
    house_station = house_lake

    # ----------------------
    # Clean up and export
    # ----------------------
    for col in ["index_right", "index_left"]:
        if col in house_station.columns:
            house_station = house_station.drop(columns=[col])
    df_links = pd.DataFrame(house_station.drop(columns="geometry"))
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_links.to_parquet(out_parquet, index=False)

    print("Wrote:", out_parquet)
    print("Columns:", df_links.columns.tolist())
    print(df_links.head())

    return df_links   