import pandas as pd

class StaticLandslideDataset:
    def __init__(self, static_terrain, ):
        self.static_terrain = static_terrain

    def load(self):
        df = pd.read_csv(self.path)
        return df