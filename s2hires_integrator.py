from csv_collector import CSVCollector
import pandas as pd
import os
from hires1_splitter import Hires1Splitter
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class S2HiresIntegrator:
    def __init__(self, s2_path, hires_path, scene_home):
        self.s2_path = s2_path
        self.hires_path = hires_path
        self.scene_home = scene_home
        scene_name = os.path.basename(self.scene_home)
        parent = os.path.dirname(self.scene_home)
        integrated_folder = f"{scene_name}_hires"
        self.integrated_path = os.path.join(parent,integrated_folder)
        self.paths = CSVCollector.collect(self.integrated_path)
        self.geo_columns = ["lon", "lat", "when"]
        self.spatial_columns = ["row", "column"]

    def process(self):
        if os.path.exists(self.integrated_path):
            print(f"Dir exists {self.integrated_path} - skipping")
            return CSVCollector.collect(self.integrated_path)
        os.mkdir(self.integrated_path)
        s2 = CSVCollector.collect(self.s2_path)
        hir = CSVCollector.collect(self.hires_path)
        s2_complete = pd.read_csv(s2["complete"])
        hires_complete = pd.read_csv(hir["complete"])

        s2_complete.sort_values(["lon", "lat", "when"], inplace=True)
        s2_complete.reset_index(inplace=True)
        hires_complete.sort_values(["lon", "lat", "when"], inplace=True)
        hires_complete.reset_index(inplace=True)
        hires_complete = hires_complete[["B04", "B03", "B02", "B08"]]
        hires_complete.columns = ["red", "green", "blue", "nir"]
        df = pd.concat([s2_complete, hires_complete], axis=1)
        df.to_csv(self.paths["complete"], index=False)

        self.aggregate()
        self.make_ml_ready()

        for ml, spl in CSVCollector.ml_split_combinations():
            s = Hires1Splitter(self.paths["ag"], split_strat=spl)
            train, test = s.split_it()

            train_len = len(train)

            train_key = CSVCollector.get_key_spatial(spl, "train", False)
            train.to_csv(self.paths[train_key], index=False)

            test_key = CSVCollector.get_key_spatial(spl, "test", False)
            test.to_csv(self.paths[test_key], index=False)

            combo = pd.concat([train, test])
            combo = self.make_ml_ready_df(combo)

            train = combo[0:train_len]
            train_key = CSVCollector.get_key_spatial(spl, "train", True)
            train.to_csv(self.paths[train_key], index=False)

            test = combo[train_len:]
            test_key = CSVCollector.get_key_spatial(spl, "test", True)
            test.to_csv(self.paths[test_key], index=False)

        return CSVCollector.collect(self.integrated_path)

    def make_ml_ready(self):
        df = pd.read_csv(self.paths["ag"])
        df = self.make_ml_ready_df(df)
        df.to_csv(self.paths["ml"], index=False)

    def make_ml_ready_df(self, df):
        df.drop(inplace=True, columns=self.spatial_columns, axis=1)
        for col in self.geo_columns:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        data = df.to_numpy()
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        df = pd.DataFrame(data=data, columns=df.columns)
        return df

    def aggregate(self):
        df = pd.read_csv(self.paths["complete"])
        df.drop(columns=self.geo_columns, axis=1, inplace=True)
        columns_to_agg = df.columns.drop(self.spatial_columns)
        df = df.groupby(self.spatial_columns)[columns_to_agg].mean().reset_index()
        df.to_csv(self.paths["ag"], index=False)