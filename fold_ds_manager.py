from soil_dataset import SoilDataset
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch


class FoldDSManager:
    def __init__(self, csv, folds=10, x=None, y=None):
        torch.manual_seed(0)
        df = pd.read_csv(csv)
        self.x = x
        self.y = y
        if y is None:
            self.y = "som"
        if x is None:
            self.x = list(df.columns)
            self.x.remove(self.y)
        self.folds = folds
        columns = self.x + [self.y]
        df = df[columns]
        self.full_data = df.to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield SoilDataset(train_data), \
                SoilDataset(test_data)

    def get_folds(self):
        return self.folds


if __name__ == "__main__":
    d = FoldDSManager("data/ml.csv")

    for fold_number, (train_ds, test_ds) in enumerate(d.get_k_folds()):
        dataloader = DataLoader(train_ds, batch_size=2, shuffle=True)
        for (x, y) in dataloader:
            print(x.shape)
            print(y.shape)
            exit(0)