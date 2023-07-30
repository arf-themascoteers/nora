from soil_dataset import SoilDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch


class DSManager:
    def __init__(self, name=None, folds=10, config=None):
        torch.manual_seed(0)
        train_df, test_df = self.get_random_train_test_df()
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
            elif config == "upper-vis":
                self.x = DSManager.get_upper_vis_bands()
            elif config == "upper-vis-props":
                self.x = DSManager.get_soil_props_upper_vis()
            elif config == "bands":
                self.x = DSManager.get_bands()
            elif config == "all":
                self.x = DSManager.get_all()

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
        return ["B02", "B03", "B04"]

    @staticmethod
    def get_upper_vis_bands():
        return ["B05", "B06", "B07", "B11", "B12", "B8A"]

    @staticmethod
    def get_soil_props_upper_vis():
        return DSManager.get_soil_props() + DSManager.get_upper_vis_bands()

    @staticmethod
    def get_soil_props():
        return ["elevation", "moisture", "temp"]

    @staticmethod
    def get_soil_props_vis():
        return DSManager.get_soil_props() + DSManager.get_vis_bands()

    @staticmethod
    def get_bands():
        columns = list(DSManager.read_from_csv().columns)
        columns = list(set(columns).difference(DSManager.get_soil_props()))
        columns.remove("som")
        return columns

    @staticmethod
    def get_all():
        columns = list(DSManager.read_from_csv().columns)
        columns.remove("som")
        return columns

    def get_random_train_test_df(self):
        df = DSManager.read_from_csv()
        return model_selection.train_test_split(df, test_size=0.2, random_state=2)

    @staticmethod
    def read_from_csv():
        DEV = False
        if DEV:
            file = "data/small_ml.csv"
        else:
            file = "data/ml.csv"
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