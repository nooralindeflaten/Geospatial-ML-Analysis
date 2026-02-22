from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional, Any

IMPACT_FLAG = {"Ja": 1, "Nei": 0, "Ukjent": 0, None: 0}

W_REGSTATUS = {
    "Godkjent kvalitet A": 1.00,
    "Registrert og godkjent": 0.90,
    "Godkjent kvalitet B": 0.85,
    "Godkjent kvalitet C": 0.70,
    "Registrert": 0.60,
}
W_NOYPOS = {
    "Eksakt": 1.00,
    "10 m": 0.98,
    "50 m": 0.93,
    "100 m": 0.90,
    "200 m": 0.87,
    "250 m": 0.85,
    "500 m": 0.80,
    "1000 m": 0.75,
    "2000 m": 0.65,
    "5000 m": 0.50,

    # unknown-ish / not registered
    "Ikke reg": 0.80,
    "Usikkert": 0.75,
    None: 0.80,
}
W_NOYTIME = {
    "Eksakt": 1.00,

    # minutes
    "1 min": 0.99,
    "5 min": 0.98,
    "10 min": 0.97,
    "15 min": 0.96,
    "30 min": 0.95,

    # hours
    "1 time": 0.92,
    "4 timer": 0.88,
    "6 timer": 0.86,
    "12 timer": 0.82,

    # days
    "1 dag": 0.78,
    "2 dager": 0.74,
    "4 dager": 0.70,
    "8 dager": 0.66,
    "16 dager": 0.62,
    "64 dager": 0.55,
    "128 dager": 0.50,

    # months / years
    "6 maneder": 0.40,
    "1 ar": 0.35,
    "2 ar": 0.30,
    "5 ar": 0.25,
    "10 ar": 0.20,
    "50 ar": 0.15,

    # unknown-ish
    "Ukjent nar pa dagen": 0.80,
    "Ukjent nar pa aret": 0.70,
    "Ukjent dato og tidspunkt": 0.65,
    "Ikke registrert": 0.80,
    None: 0.80,
}
IMPACT_FLAG = {"Ja": 1, "Nei": 0, "Ukjent": 0, None: 0}
w_q = W_REGSTATUS.get('regStatus', 0.75) * W_NOYPOS.get('noyPosisjon', 0.80) * W_NOYTIME.get('noySkredTidspunkt', 0.80)
w_q = max(0.4, min(1.0, w_q))  # clamp

damage_flag = max(IMPACT_FLAG.get('bygnSkadet', 0),
                  IMPACT_FLAG.get('vegSkadet', 0),
                  IMPACT_FLAG.get('baneSkadet', 0),
                  IMPACT_FLAG.get('skogJordDyrSkadet', 0),
                  IMPACT_FLAG.get('annetSkadet', 0))

w_imp = 1.0 + 0.25 * damage_flag
w_event = w_q * w_imp



class LandslideLabels:
    def __init__(self,region,base_dir):
        self.region = region
        self.base_dir = base_dir
        self.hazard_package = self.base_dir / "hazards" / f"{self.region}_hazards.gpkg"
        self.hydro_package  = self.base_dir / "hydro"   / f"{self.region}_hydro.gpkg"
        self.house_package  = self.base_dir / "houses"  / f"houses_{self.region}.gpkg"
        self._regstatus_score = {
            "Godkjent kvalitet A": 4,
            "Godkjent kvalitet B": 3,
            "Godkjent kvalitet C": 2,
            "Registrert og godkjent": 2,
            "Registrert": 1,
        }

        self._pos_to_m = {
            "Eksakt": 0,
            "10 m": 10,
            "50 m": 50,
            "100 m": 100,
            "200 m": 200,
            "250 m": 250,
            "500 m": 500,
            "1000 m": 1000,
            "2000 m": 2000,
            "5000 m": 5000,
            "Usikkert": np.nan,
            "Ikke reg": np.nan,
            None: np.nan,
        }

        self.landslide_cols_keep = [
            "skredID", "skredType", "skredNavn", "stedsnavn",
            "skredTidspunkt", "noySkredTidspunkt", "noyPosisjon",
            "regStatus", "kommunenummer", "fylkesnummer",
            "bygnSkadet", "vegSkadet", "baneSkadet",
            "evakuering", "redningsaksjon",
            "vaerObservasjon", "beskrivelse",
            "skogJordDyrSkadet", "fremkomstmiddelSkadet", "annetSkadet",
            "SHAPE_Length", "utlopUtlosningOmrID",
        ]
        
    def _pos_m(self,series: pd.Series):
    # robust to weird strings
        s = series.astype("object")
        return s.map(self._pos_to_m).astype("float")

    def _reg_score(self,series: pd.Series) -> pd.Series:
        s = series.astype("object")
        return s.map(self._regstatus_score).fillna(0).astype("int")

    def read_layer(self,gpkg_path,layer,cols_keep):
        gdf = gpd.read_file(gpkg_path,layer=layer).to_crs(25833)
        if cols_keep is not None:
            cols = [c for c in cols_keep if c in gdf.columns]
            cols = cols + (["geometry"] if "geometry" in gdf.columns else [])
            gdf = gdf[cols]

        return gdf
    
    def read_hazard_layers(self):
        """
        Reads landslide layers for region and returns dict of GeoDataFrames.
        """
        layers = {
            "landslides": "landslides",
            "trigger_points": "landslide_trigger_points",
            "runout_points": "landslide_runout_points",
            "source_areas": "landslide_source_areas",
            "runout_areas": "landslide_runout_areas",
        }

        out = {}
        for k, layer_name in layers.items():
            if k == "landslides":
                out[k] = self.read_layer(self.hazard_package, layer_name, cols_keep=None)
            else:
                out[k] = self.read_layer(self.hazard_package, layer_name, cols_keep=None)
        return out
    
    def _filter_by_id(gdf,id_col,skred_id):
        if id_col in gdf.columns:
            return gdf[gdf[id_col] == skred_id].copy()
        return gdf.iloc[0:0].copy()
    
    
    def _spatial_fallback_within(gdf,geom,buffer_m):
        """
        Spatial fallback: return rows whose geometry intersects geom buffered by buffer_m.
        """
        if gdf.empty:
            return gdf
        buf = geom.buffer(buffer_m)
        return gdf[gdf.geometry.intersects(buf)].copy()
    
    def _spatial_fallback_nearest(gdf,geom,k=10,max_dist_m=None):
        """
        Spatial fallback: take k nearest geometries to geom (optionally within max_dist_m).
        """
        if gdf.empty:
            return gdf
        d = gdf.geometry.distance(geom)
        if max_dist_m is not None:
            gdf = gdf[d <= max_dist_m].copy()
            if gdf.empty:
                return gdf
            d = gdf.geometry.distance(geom)
        return gdf.loc[d.nsmallest(k).index].copy()
    
    def events_master(self,landslides):
        keep_cols = [c for c in self.landslide_cols_keep if c in landslides.columns]
        keep_cols = list(dict.fromkeys(keep_cols)) 

        df = landslides[keep_cols].copy()
        df["reg_score"] = self._reg_score(df.get("regStatus", pd.Series(index=df.index, dtype="object")))
        df["pos_m"] = self._pos_m(df.get("noyPosisjon", pd.Series(index=df.index, dtype="object")))

        sort_cols = ["reg_score", "pos_m"]
        asc = [False, True]
        if "skredTidspunkt" in df.columns:
            sort_cols.append("skredTidspunkt")
            asc.append(False)

        df = df.sort_values(sort_cols, ascending=asc)
        df_best = df.drop_duplicates("skredID", keep="first").reset_index(drop=True)

        return df_best
    
    def bundle_by_skredID(
        self,
        id_col="skredID",
        buffer_m=250.0,
        nearest_k=25,
        nearest_max_dist_m=2000.0):
        """
        Returns:
        {
          skredID: {
            "landslide": GeoDataFrame (point(s)),
            "source_areas": GeoDataFrame (polygons),
            "runout_areas": GeoDataFrame (polygons),
            "trigger_points": GeoDataFrame (points),
            "runout_points": GeoDataFrame (points),
          },
          ...
        }

        Strategy:
        - Primary: link by skredID (attribute join)
        - Fallback: if a layer has missing/empty skredID match, use spatial relation:
            * polygons: within/near the event point
            * points: within a buffer of event point; optionally nearest k within a max distance
        """
        layers = self.read_hazard_layers()
        slides = layers["landslides"].copy()

        if id_col not in slides.columns:
            raise ValueError(f"`{id_col}` not found in landslides layer. Available: {list(slides.columns)}")

        # Keep only rows with skredID (drop null IDs)
        slides = slides[slides[id_col].notna()].copy()

        bundles: Dict[str, Dict[str, gpd.GeoDataFrame]] = {}

        for skred_id, slide_gdf in slides.groupby(id_col):
            # If you can have multiple event points per skredID, keep them.
            slide_gdf = slide_gdf.copy()

            # For spatial fallback, use a representative geometry (centroid of all points)
            # If multiple points, unary_union gives MultiPoint; centroid is ok.
            rep_geom = slide_gdf.unary_union.centroid

            # 1) Polygons (source + runout)
            src_poly = self._filter_by_id(layers["source_areas"], id_col, skred_id)
            run_poly = self._filter_by_id(layers["runout_areas"], id_col, skred_id)

            # fallback: if polygons missing, grab nearby polygons (rare but happens)
            if src_poly.empty:
                src_poly = self._spatial_fallback_nearest(
                    layers["source_areas"], rep_geom, k=nearest_k, max_dist_m=nearest_max_dist_m
                )
            if run_poly.empty:
                run_poly = self._spatial_fallback_nearest(
                    layers["runout_areas"], rep_geom, k=nearest_k, max_dist_m=nearest_max_dist_m
                )

            # 2) Points (trigger + runout)
            trig_pts = self._filter_by_id(layers["trigger_points"], id_col, skred_id)
            run_pts  = self._filter_by_id(layers["runout_points"], id_col, skred_id)

            # fallback: if missing ID match, use polygons if available (best), else buffer around event point
            if trig_pts.empty:
                if not src_poly.empty:
                    trig_pts = layers["trigger_points"][layers["trigger_points"].geometry.within(src_poly.unary_union)].copy()
                if trig_pts.empty:
                    trig_pts = self._spatial_fallback_within(layers["trigger_points"], rep_geom, buffer_m=buffer_m)

            if run_pts.empty:
                if not run_poly.empty:
                    run_pts = layers["runout_points"][layers["runout_points"].geometry.within(run_poly.unary_union)].copy()
                if run_pts.empty:
                    run_pts = self._spatial_fallback_within(layers["runout_points"], rep_geom, buffer_m=buffer_m)

            bundles[str(skred_id)] = {
                "landslide": slide_gdf,
                "source_areas": src_poly,
                "runout_areas": run_poly,
                "trigger_points": trig_pts,
                "runout_points": run_pts,
            }

        return bundles
    
    
    
    
class HousePatches:
    def __init__(self, region,base_dir,patches_m=200):
        self.base_dir = Path(base_dir)
        self.region = region
        self.houses_gdf = gpd.read_file(self.base_dir / f"raw/vector/houses/houses_{self.region}.gpkg",layer=f"houses_{region}").to_crs(25833)
        self.patches_m = patches_m
        self.processed_raster_dir = self.base_dir / "processed/rasters"
        self.raw_raster_dir =  self.base_dir / "raw/rasters"
        self.hazard_package = self.base_dir / "hazards" / f"{self.region}_hazards.gpkg"

    
    
    def raster_paths(self):
        return {
        "dtm": self.raw_raster_dir / f"{self.region}_dtm10.tif",
        "slope": self.processed_raster_dir / f"{self.region}_slope_deg.tif",
        "aspect": self.processed_raster_dir / f"{self.region}_aspect_deg.tif",
        "curv": self.processed_raster_dir / f"{self.region}_curvature.tif",
        "flowacc": self.processed_raster_dir / f"{self.region}_flowacc_d8.tif",
        "twi": self.processed_raster_dir / f"{self.region}_twi.tif",
    }
        
        
    def sample_raster_at_points(self,raster_paths, gdf):
        """
        Sample a single-band raster at point locations.

        - raster_path: path to .tif
        - gdf: GeoDataFrame with point geometries
        - out_col: name of output column with sampled values
        """
        keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
        patches = {k: rasterio.open(raster_paths[k]) for k in keys}
        for k in keys:
# for idx, row in tqdm(gdf.iterrows(), total=len(gdf)): might add this? 
            src = patches[k]
            raster_crs = src.crs
            nodata = src.nodata

            # Reproject houses if needed
            if gdf.crs != raster_crs:
                print(f"Reprojecting from {gdf.crs} to {raster_crs}")
                gdf = gdf.to_crs(raster_crs)

            # Extract coordinates
            coords = [(geom.x, geom.y) for geom in gdf.geometry]

            # Sample raster
            print(f"Sampling {len(coords)} points from a region ...")
            sampled = list(src.sample(coords))

        # Flatten results (each sample is e.g. [value])
            values = [s[0] if s is not None else np.nan for s in sampled]
            # Replace nodata with NaN
            if nodata is not None:
                values = [np.nan if v == nodata else v for v in values]
            
            # Assign to gdf
            gdf[f"{k}"] = values
                
        return gdf
    
    def land_slide_links(self, l_links,base_dir="../", region="sogn"):
        """ 
        we get simple landslide features for each house here,
        what's missing is we need to take the closest landslide event and use this to add weights or 
        we do this later for a combined score for each landslide within house? 
        
        """
        houses_gdf = self.houses_gdf.copy()
        events_gpkg = Path(base_dir) / "raw" / "vector" / "hazards" / f"{region}_hazards.gpkg"
        landslides = gpd.read_file(events_gpkg, layer="landslides").to_crs(epsg=25833)
        l_runout_points = gpd.read_file(events_gpkg, layer="landslide_trigger_points").to_crs(epsg=25833)
        l_source_points = gpd.read_file(events_gpkg, layer="landslide_runout_points").to_crs(epsg=25833)
        runout = gpd.read_file(
                    events_gpkg,
                    layer="landslide_runout_areas"
                ).to_crs(25833)
        
        l_union = landslides.union_all()
        l_runout_points_union = l_runout_points.union_all()
        l_source_points_union = l_source_points.union_all()
        
        houses_gdf['dist_landslide_m'] = houses_gdf.geometry.apply(lambda x: l_union.distance(x))
        houses_gdf['dist_landslide_runout_m'] = houses_gdf.geometry.apply(lambda x: l_runout_points_union.distance(x))
        houses_gdf['dist_landslide_source_m'] = houses_gdf.geometry.apply(lambda x: l_source_points_union.distance(x))
        buffered = runout.copy()
        buffered["geometry"] = buffered.geometry.buffer(20)

        # label = 1 if house is inside buffered runout
        houses_gdf["y_runout_hit"] = houses_gdf.geometry.within(buffered.unary_union).astype(int)
        
        # count number of times a house is listed in the l_links df and set as feature n_landslides_within_100m
        houses_gdf["n_landslides_within_100m"] = houses_gdf["bygningsnummer"].map(
            l_links["bygningsnummer"].value_counts()
        ).fillna(0).astype(int)
        return houses_gdf
    
    def keep_only(df: pd.DataFrame, keep: list[str], *, always_keep: str | None = None) -> pd.DataFrame:
        keep_set = set(keep)
        if always_keep:
            keep_set.add(always_keep)

        cols = [c for c in df.columns if c in keep_set]
        return df.loc[:, cols].copy()

    def get_hyd_features(self):
        """ 
        Get catchment area related to houses. extract hydrological features for 
        
        """
        catch_df = pd.read_parquet(self.base_dir / f"raw/links/{self.region}/{self.region}_catchment_main.parquet")
        l_links = pd.read_parquet(self.base_dir / f"processed/links/landslide_links_{self.region}.parquet")
        clean_landslide = self.land_slide_links(l_links, base_dir=self.base_dir, region=self.region)
        # if both catch_df and clean_landslide have same columns, merge them and keep catch_df columns
        merged = catch_df.merge(
            clean_landslide,
            on="bygningsnummer",
            how="left",
        )
        if 'index_right' in merged.columns:
            merged = merged.drop(columns=['index_right'])
        return merged
    
    def baseline_ds(self):
        hyd_keep_baseline = [
                    "arealEnhet_km2",
                    "arealTotal_km2",
                    "QNormal_lskm2",
                    "QNormal_Mm3Aar",
                    "QNormalOppstrm_Mm3Aar",
                    "elveordenstrahler",
                    "lengde_m",
                    "dist_to_river",
                    "dist_to_lake",
                    "dist_to_hyd",
                    "hoyde",
                    "areal_km2",
                    "overordnetVassdragsNr",
                    "nedborfeltVassdragsNr",
                    "vassdragsomradeNr",
                    "vassdragsNr"
                ]
        leakage_cols = [
        # derived from landslide geometries / "closest" tables
        "inside_source_area", "inside_runout_area", "any_landslide_area_inside",
        "dist_to_trigger_point", "dist_to_runout_point",
        "dist_to_source_area", "dist_to_runout_area",
        "dist_to_landslide_event",
        "sourcearea_area_m2", "runoutarea_area_m2",
        "trigger_point_far", "runout_point_far", "landslide_event_far",
        # old target / old hack
        "landslide_hazard_level",
        ]
        raster_paths = self.raster_paths()
        terrain_df = self.sample_raster_at_points(raster_paths, self.houses_gdf.copy())

        merged = self.get_hyd_features()
        merged = merged.merge(
            terrain_df[["bygningsnummer", "dtm", "slope", "aspect", "curv", "flowacc", "twi"]],
            on="bygningsnummer",
            how="left",
        )

        terrain_cols = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]

        # ADD your simple landslide-derived numeric features
        landslide_simple_cols = [
            "dist_landslide_m",
            "dist_landslide_runout_m",
            "dist_landslide_source_m",
            "n_landslides_within_100m",
        ]

        keep = [c for c in (hyd_keep_baseline + terrain_cols + landslide_simple_cols) if c in merged.columns]

        # hard safety: drop leakage if it somehow exists in keep
        keep = [c for c in keep if c not in leakage_cols]

        # optional: keep ids/geometry so you can build groups/splits
        base_cols = [c for c in ["bygningsnummer", "geometry", "y_runout_hit"] if c in merged.columns]
        merged_out = merged[base_cols + keep].copy()
        merged_geo = gpd.GeoDataFrame(merged_out, geometry="geometry",crs=25833)
        return merged_out, keep, merged_geo
        




def build_ds_continous_target(df: pd.DataFrame, target_col: str = "landslide_hazard_level") -> pd.DataFrame:
    """
    Build continuous target variable for landslide hazard level.

    Mapping:
    - 0 (No hazard) -> 0.0
    - 1 (Low hazard) -> 0.33
    - 2 (Medium hazard) -> 0.66
    - 3 (High hazard) -> 1.0

    Args:
        df: Input DataFrame with categorical target column.
        target_col: Name of the target column.

    Returns:
        DataFrame with new continuous target column 'landslide_hazard_continuous'.
    """
    mapping = {
        0: 0.0,
        1: 0.33,
        2: 0.66,
        3: 1.0,
    }
    df = df.copy()
    df["landslide_hazard_continuous"] = df[target_col].map(mapping).astype("float")
    return df










bygningsnummer_COL = 'bygningsnummer'

def build_house_event_links(region,events_gpkg: Path, events_layer: str, house_buffer_m: float) -> gpd.GeoDataFrame:
    houses = gpd.read_file(f"geodata_analysis_final_version/data/raw/vector/exposure/{region}_houses.gpkg", layer=f"houses_{region}").to_crs(25833)
    houses = houses.set_geometry('geometry')
    events = gpd.read_file(events_gpkg, layer=events_layer).to_crs(25833)
    # Buffer houses
    houses_buf = houses[['bygningsnummer', "geometry"]].copy()
    houses_buf["geometry"] = houses_buf.geometry.buffer(house_buffer_m)

    # 2) CRS sanity
    if houses.crs is None:
        raise RuntimeError("Houses layer has no CRS set")

    if houses.crs.to_string() != events.crs.to_string():
        houses = houses.to_crs(events.crs)

    # 3) Make house point geometry explicit
    houses = houses.rename_geometry("house_geom")

    if bygningsnummer_COL not in houses.columns:
        raise RuntimeError(f"{bygningsnummer_COL} not found in houses columns")

    # 4) Create buffered houses for intersection test
    houses_buf = houses[[bygningsnummer_COL, "house_geom"]].copy()
    houses_buf["geometry"] = houses_buf["house_geom"].buffer(house_buffer_m)
    houses_buf = gpd.GeoDataFrame(houses_buf, geometry="geometry", crs=events.crs)

    # 5) Spatial join: which events fall inside each house buffer?
    joined = gpd.sjoin(
        houses_buf,
        events,         # uses event_geom as geometry
        how="inner",
        predicate="intersects",
    )

    # After sjoin:
    # - 'geometry' = house buffer (left geometry)
    # - 'house_geom' = house point
    # - 'event_geom' = event point (still a column)

    # 6) Compute distance from house point to event point
    joined["distance_m"] = joined.apply(
        lambda row: row["house_geom"].distance(row["geometry"]),
        axis=1,
    )

    # 7) Build a *clean* table, keeping only one geometry column (house point)
    cols_keep = [
        bygningsnummer_COL,
        "skredID",
        "distance_m",
        "skredType",
        "skredTidspunkt",
        "year",
        "bygnSkadet",
        "vegSkadet",
        "persBerort",
        "kommunenavn",
        "kommunenummer",
        "house_geom",  # this will be our only geometry column
    ]
    cols_keep = [c for c in cols_keep if c in joined.columns]

    links = joined[cols_keep].copy()

    # 8) Set active geometry = house point and rename to standard 'geometry'
    links = gpd.GeoDataFrame(links, geometry="house_geom", crs=events.crs)
    links = links.rename_geometry("geometry")

    # Now there is only ONE geometry column: 'geometry' (house point).
    # 'event_geom' and the buffer geometry are NOT included anymore, so no geometry conflict.

    # 9) Save to file
    