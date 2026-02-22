import torch
from pathlib import Path
from src.utils.paths import ProjectPaths,RegionContext, GlobalContext, RasterPaths


import numpy as np
import pandas as pd
from torch_geometric.data import Data
from src.utils.houses import Houses
from src.utils.loaders import aggregate_per_house, load_houses_data, load_graph_bundle


import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import geopandas as gpd
import torch
import rasterio
from rasterio.windows import Window
import torch_geometric.nn

def build_knn_edge_index(coords: np.ndarray, k: int = 8):
    """
    coords: [N, 2] array of x,y in the same CRS (EPSG:25833)
    returns edge_index: [2, E]
    """

    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), algorithm="auto")
    nbrs.fit(coords)
    dists, idxs = nbrs.kneighbors(coords)

    # idxs includes self in position 0
    src_list = []
    dst_list = []

    for i in range(len(coords)):
        neigh = idxs[i][1:]  # remove self
        for j in neigh:
            src_list.append(i)
            dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index

def build_data_objects(
    region,
    house_id_col="bygningsnummer",
    base_dir="../",
    flood_label="hazard_class_flom",
    landslide_label="landslide_hazard_level",
    graph_path=None,
):
    houses_gdf = load_houses_data(
        region,
        base_dir=base_dir,
    )
    houses_gdf = houses_gdf.set_geometry("geometry").copy()
    houses_gdf[house_id_col] = houses_gdf[house_id_col].astype(int)

    graph = load_graph_bundle(graph_path)
    graph_ids = graph["house_id"].cpu().numpy().tolist()

    # 1) filter houses to graph ids
    houses = houses_gdf[houses_gdf[house_id_col].isin(graph_ids)].copy()

    # 2) enforce EXACT order = graph house_id order
    order_map = {hid: i for i, hid in enumerate(graph_ids)}
    houses["__order"] = houses[house_id_col].map(order_map)
    houses = houses.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # sanity
    assert len(houses) == len(graph_ids), f"Graph nodes ({len(graph_ids)}) != houses ({len(houses)})"

    # 3) build numeric features
    df_for_features = pd.DataFrame(houses.drop(columns=["geometry"], errors="ignore"))

    drop_always = {house_id_col, "bygningId", "gml_id", "uuidBygning", "versjonId"}
    drop_labels = {flood_label, landslide_label}

    num_df = df_for_features.select_dtypes(include=[np.number]).copy()
    num_df = num_df[[c for c in num_df.columns if c not in (drop_always | drop_labels)]]
    num_df = num_df.dropna(axis=1, how="all")
    num_df = num_df.fillna(num_df.median(numeric_only=True))

    x = torch.tensor(num_df.to_numpy(dtype=np.float32))

    # 4) labels + masks
    y_flood_t = None
    y_landslide_t = None
    y_flood_mask_t = None
    y_landslide_mask_t = None

    if flood_label in df_for_features.columns:
        y_f = df_for_features[flood_label].to_numpy()
        flood_mask = ~pd.isna(y_f)
        y_f = pd.Series(y_f).fillna(-1).astype(int).to_numpy()
        y_flood_t = torch.tensor(y_f, dtype=torch.long)
        y_flood_mask_t = torch.tensor(flood_mask, dtype=torch.bool)

    if landslide_label in df_for_features.columns:
        y_l = df_for_features[landslide_label].to_numpy()
        landslide_mask = ~pd.isna(y_l)
        y_l = pd.Series(y_l).fillna(-1).astype(int).to_numpy()
        y_landslide_t = torch.tensor(y_l, dtype=torch.long)
        y_landslide_mask_t = torch.tensor(landslide_mask, dtype=torch.bool)

    data = Data(
        x=x,
        pos=graph["pos"],
        edge_index=graph["edge_index"],
        house_id=graph["house_id"],
    )

    if y_flood_t is not None:
        data.y_flood = y_flood_t
        data.y_flood_mask = y_flood_mask_t

    if y_landslide_t is not None:
        data.y_landslide = y_landslide_t
        data.y_landslide_mask = y_landslide_mask_t

    return data, houses, num_df.columns.tolist(), graph

from torch_geometric.transforms import RandomNodeSplit

def add_splits(data, val_ratio=0.1, test_ratio=0.1):
    splitter = RandomNodeSplit(
        num_val=int(val_ratio * data.num_nodes),
        num_test=int(test_ratio * data.num_nodes),
    )
    data = splitter(data)
    return data

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import geopandas as gpd
import torch
import rasterio
from rasterio.windows import Window
import torch_geometric.nn

def build_knn_edge_index(coords: np.ndarray, k: int = 8):
    """
    coords: [N, 2] array of x,y in the same CRS (EPSG:25833)
    returns edge_index: [2, E]
    """

    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), algorithm="auto")
    nbrs.fit(coords)
    dists, idxs = nbrs.kneighbors(coords)

    # idxs includes self in position 0
    src_list = []
    dst_list = []

    for i in range(len(coords)):
        neigh = idxs[i][1:]  # remove self
        for j in neigh:
            src_list.append(i)
            dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


from pathlib import Path
import torch
import numpy as np

def build_and_save_knn_graph(
    houses_gdf,
    out_path,
    house_id_col="bygningsnummer",
    k=8,
):
    # 1) canonical ordering
    
    houses = houses_gdf[[house_id_col, "geometry"]].copy()
    houses[house_id_col] = houses[house_id_col].astype(int)
    houses = houses.dropna(subset=["geometry"])
    houses = houses.sort_values(house_id_col).reset_index(drop=True)

    # 2) positions
    coords = np.array([(g.x, g.y) for g in houses.geometry], dtype=np.float32)
    pos = torch.from_numpy(coords)

    # 3) edge index
    edge_index = build_knn_edge_index(coords, k=k)
    # 4) ids
    house_id = torch.tensor(houses[house_id_col].to_numpy(), dtype=torch.long)

    payload = {
        "house_id": house_id,
        "pos": pos,
        "edge_index": edge_index,
        "k": k,
        "house_id_col": house_id_col,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print("Saved graph:", out_path, "| nodes:", len(houses))

    return payload