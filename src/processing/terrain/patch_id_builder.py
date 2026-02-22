import geopandas as gpd
import pandas as pd
from pathlib import Path

def build_patch_index(
    houses_gpkg: str | Path,
    houses_layer: str,
    label_col: str = "hazard_class_flom",
    flood_p : str | Path = None,
    id_col: str = "bygningsnummer",
    target_crs: str = "EPSG:25833",
    val_frac: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    out_index_parquet: str | Path | None = None,
):
    """
    Returns a DataFrame with at least:
      [id_col, label_col, x, y, split]
    """
    houses = gpd.read_file(houses_gpkg, layer=houses_layer)

    if houses.crs is None or houses.crs.to_string() != target_crs:
        houses = houses.to_crs(target_crs)

    # Keep only labeled houses
    df_links = pd.read_parquet(flood_p)
    houses = houses.merge(df_links, on=id_col, how="left")
    houses = houses[houses[label_col].notna()].copy()

    # Normalize ID
    if id_col in houses.columns:
        try:
            houses[id_col] = houses[id_col].astype(int)
        except Exception:
            pass
    else:
        raise ValueError(f"Missing id_col='{id_col}' in houses layer.")

    # Extract XY for fast access later (avoid geometry in parquet)
    houses["x"] = houses.geometry.x
    houses["y"] = houses.geometry.y

    df = pd.DataFrame(houses[[id_col, label_col, "x", "y"]]).reset_index(drop=True)
    # Split
    n = len(df)
    val_n = int(n * val_frac)

    if val_n == 0:
        df["split"] = "train"
        if out_index_parquet:
            Path(out_index_parquet).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_index_parquet, index=False)
        return df

    if stratify:
        # simple stratified split without sklearn dependency
        df["split"] = "train"
        for cls, group in df.groupby(label_col):
            k = max(1, int(len(group) * val_frac))
            val_idx = group.sample(n=k, random_state=random_state).index
            df.loc[val_idx, "split"] = "val"
    else:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        df["split"] = "train"
        df.loc[:val_n - 1, "split"] = "val"

    if out_index_parquet:
        out_index_parquet = Path(out_index_parquet)
        out_index_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_index_parquet, index=False)

    return df