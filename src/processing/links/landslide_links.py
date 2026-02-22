import geopandas as gpd
import pandas as pd

def landslide_links(houses_gdf, hazard_gpkg, out_path):
    landslide_gdf = gpd.read_file(hazard_gpkg, layer="landslides").to_crs(25833)

    # iterate over houses and see how many landslides are within 100 m
    records = []
    for idx, house in houses_gdf.iterrows():
        geom = house.geometry
        if geom is None or geom.is_empty:
            continue

        nearby_landslides = landslide_gdf[landslide_gdf.geometry.distance(geom) <= 100]

        records.append({
            **house,
            "num_landslides_100m": len(nearby_landslides),
        })
    df = pd.DataFrame(records)
    df.drop(columns="geometry", inplace=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved landslide links for {len(df)} houses to {out_path}")