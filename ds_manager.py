from soil_dataset import SoilDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch


class DSManager:
    def __init__(self, name=None, folds=10, source=None, config=None):
        torch.manual_seed(0)
        DEV = False
        if DEV:
            self.file = "data/small_ml.csv"
        else:
            self.file = "data/ml.csv"

        train_df, test_df = self.get_random_train_test_df(source)
        self.x = []
        if config is None:
            config = "all"

        if isinstance(config,str):
            if config == "vis":
                self.x = DSManager.get_vis_bands()
            elif config == "props":
                self.x = DSManager.get_soil_props()
            elif config == "vis-props":
                self.x = DSManager.get_soil_props_vis()
            elif config == "bands":
                self.x = list(train_df.columns)
                self.x = list(set(self.x).difference(DSManager.get_soil_props()))
                self.x.remove("som")
            elif config == "all":
                self.x = list(train_df.columns)
                self.x.remove("som")

        elif type(config) == list:
            self.x = config

        self.y = "som"
        self.name = name
        self.folds = folds

        df = pd.concat([train_df, test_df])
        columns = self.x + [self.y]
        df = df[columns]
        self.full_data = df.to_numpy()
        self.full_data = self._normalize(self.full_data)
        self.train = self.full_data[0:len(train_df)]
        self.test = self.full_data[len(train_df):]

    @staticmethod
    def get_vis_bands():
        return ["B02_1", "B03_1", "B04_1"]

    @staticmethod
    def get_soil_props():
        return ["elevation", "moisture", "temp"]

    @staticmethod
    def get_soil_props_vis():
        return DSManager.get_soil_props() + DSManager.get_vis_bands()

    def get_random_train_test_df(self, source):
        df = self.read_from_csv(self.file)
        if source is not None:
            df = df[df["source"] == source]
        return model_selection.train_test_split(df, test_size=0.2, random_state=2)

    def read_from_csv(self, file):
        df = pd.read_csv(file)
        return df

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

    def get_updated_columns(self, df, columns):
        indices = []
        for index,col in enumerate(df.columns):
            if col in columns:
                indices.append(index)
            else:
                for i in columns:
                    if col.startswith(f"{i}_"):
                        indices.append(index)
        return indices


if __name__ == "__main__":
    d = DSManager(config="all")

    for fold_number, (train_ds, test_ds) in enumerate(d.get_k_folds()):
        dataloader = DataLoader(train_ds, batch_size=2, shuffle=True)
        for (x, y) in dataloader:
            print(x.shape)
            print(y.shape)
            exit(0)