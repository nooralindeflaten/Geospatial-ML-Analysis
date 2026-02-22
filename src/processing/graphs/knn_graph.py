from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from tqdm import tqdm

def build_knn_edges(coords: np.ndarray, catchment_ids, k: int):
    """
    coords: [N, 2] array of (x, y) in meters
    catchment_ids: list/array of length N with catchment_id (can be str or int)
    k: number of neighbours per node

    Returns:
      edge_index (2, E) np.array of node indices
      edge_attr  (E, 2) np.array of [distance_m, same_catchment_flag]
    """
    N = coords.shape[0]
    x = coords[:, 0]
    y = coords[:, 1]

    edges = []
    attrs = []

    for i in tqdm(range(N), desc=f"Building KNN edges (k={k})"):
        dx = x - x[i]
        dy = y - y[i]
        dist2 = dx * dx + dy * dy
        dist2[i] = np.inf  # exclude self

        # indices of k nearest neighbours
        if N - 1 <= k:
            nn_idx = np.where(np.isfinite(dist2))[0]
        else:
            nn_idx = np.argpartition(dist2, k)[:k]

        for j in nn_idx:
            d = float(np.sqrt(dist2[j]))
            same_catch = 1.0 if catchment_ids[i] == catchment_ids[j] else 0.0

            # add both directions (undirected graph)
            edges.append((i, j))
            attrs.append((d, same_catch))

            edges.append((j, i))
            attrs.append((d, same_catch))

    # Deduplicate edges (in case KNN choices overlap)
    edge_array = np.array(edges, dtype=np.int64)
    attr_array = np.array(attrs, dtype=np.float32)

    # Represent undirected edge as sorted pair to deduplicate
    undirected_keys = np.sort(edge_array, axis=1)
    _, unique_idx = np.unique(undirected_keys, axis=0, return_index=True)

    edge_array = edge_array[unique_idx]
    attr_array = attr_array[unique_idx]

    # Transpose to shape (2, E) for PyG-style
    edge_index = edge_array.T  # [2, E]

    return edge_index, attr_array