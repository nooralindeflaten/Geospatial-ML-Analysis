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

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    # Loss function for combining cross entropy and mse loss
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, labels):
        for label in range(0,labels.size()[0]-1):
            if labels[:,label].sum() == -1 * labels.size()[1]:
                mse = self.mse_loss(outputs[:,label], labels[:,label].float())
                cel = self.cross_entropy(outputs[:,label], labels[:,label])
            else:
                cel = 0
                mse = 0
        return cel + mse
                 

class TerrainEncoderCNN(nn.Module):
    def __init__(self, in_channels=6, t_dim=128):
        super(TerrainEncoderCNN, self).__init__()
        import torch.nn as nn
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),)
        
        self.pool = nn.MaxPool2d(2)        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, t_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.head(x)
        return x
    
    
class TabularMLP(nn.Module):
    def __init__(self, in_dim, d_tab):
        super(TabularMLP, self).__init__()
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),

            nn.Linear(128, d_tab),
        )
    def forward(self, x):
        return self.net(x)


class CombinedMLP(nn.Module):
    def __init__(self, d_terrain=128, d_tab=64, d_node=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_terrain + d_tab, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_node),
        )

    def forward(self, z_terrain, z_tab):
        return self.net(torch.cat([z_terrain, z_tab], dim=-1))
    
class GNNBackbone(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=192, num_layers=3):
        super(GNNBackbone, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        return x
    
    
class HazardHeads(nn.Module):
    def __init__(self, in_dim, num_flood_classes=3):
        super(HazardHeads, self).__init__()
        self.flood_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_flood_classes),
        )
    def forward(self, x):
        flood_logits = self.flood_head(x)
        return flood_logits
    
    
class HazardGNNModel(nn.Module):
    def __init__(self, terrain_in_channels=6, terrain_dim=128, tabular_in_dim=50, tabular_dim=64,
                 gnn_hidden_dim=192, gnn_layers=3, num_flood_classes=3):
        super(HazardGNNModel, self).__init__()
        
        self.terrain_encoder = TerrainEncoderCNN(
            in_channels=terrain_in_channels,
            t_dim=terrain_dim
        )
        self.tabular_mlp = TabularMLP(
            in_dim=tabular_in_dim,
            d_tab=tabular_dim
        )
        self.combined_mlp = CombinedMLP(
            d_terrain=terrain_dim,
            d_tab=tabular_dim,
            d_node=gnn_hidden_dim
        )

        # ✅ FIXED: input dim matches combined output
        self.gnn_backbone = GNNBackbone(
            in_dim=gnn_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers
        )

        self.hazard_heads = HazardHeads(
            in_dim=gnn_hidden_dim,
            num_flood_classes=num_flood_classes,
        )
    def forward(self, data, patches, node_idx):
        z_terrain = self.terrain_encoder(patches)
        z_tab = self.tabular_mlp(data.x[node_idx])
        combined = self.combined_mlp(z_terrain, z_tab)

        x_full = data.x.new_zeros((data.num_nodes, combined.size(-1)))
        x_full[node_idx] = combined

        gnn_feats = self.gnn_backbone(x_full, data.edge_index)
        out = gnn_feats[node_idx]

        flood_logits = self.hazard_heads(out)
        return flood_logits

