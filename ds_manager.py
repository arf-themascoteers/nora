from soil_dataset import SoilDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch


class DSManager:
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

        train_df, test_df = model_selection.train_test_split(df, test_size=0.2, random_state=2)
        df = pd.concat([train_df, test_df])
        columns = self.x + [self.y]
        df = df[columns]
        self.full_data = df.to_numpy()
        self.full_data = self._normalize(self.full_data)
        self.train = self.full_data[0:len(train_df)]
        self.test = self.full_data[len(train_df):]

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield SoilDataset(train_data), \
                SoilDataset(test_data)

    def get_folds(self):
        return self.folds

    def _normalize(self, data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data


if __name__ == "__main__":
    d = DSManager("data/ml.csv")

    for fold_number, (train_ds, test_ds) in enumerate(d.get_k_folds()):
        dataloader = DataLoader(train_ds, batch_size=2, shuffle=True)
        for (x, y) in dataloader:
            print(x.shape)
            print(y.shape)
            exit(0)