import pandas as pd
from torch.utils.data import DataLoader
import torch
from soil_dataset import SoilDataset


class SplitDSManager:
    def __init__(self, train=None, test=None, x=None, y=None):
        torch.manual_seed(0)
        self.train_df = pd.read_csv(train)
        self.test_df = pd.read_csv(test)
        self.x = x
        self.y = y
        if y is None:
            self.y = "som"
        if x is None:
            self.x = list(self.train_df.columns)
            self.x.remove(self.y)
        columns = self.x + [self.y]
        self.train_df = self.train_df[columns]
        self.test_df = self.test_df[columns]
        self.train_df = self.train_df.sample(frac=1)
        self.test_df = self.test_df.sample(frac=1)
        self.train_np = self.train_df.to_numpy()
        self.test_np = self.test_df.to_numpy()

    def get_datasets(self):
        return SoilDataset(self.train_np), SoilDataset(self.test_np)
