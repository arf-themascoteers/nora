import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from splitter import Splitter
from csv_collector import CSVCollector


class CSVCreator:
    def __init__(self, df, export_dir, ag):
        self.df = df
        self.ag = ag
        self.base_dir = export_dir
        self.paths = CSVCollector.collect(self.base_dir)
        self.geo_columns = ["lon", "lat", "when"]
        self.spatial_columns = ["scene", "row", "column"]

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
        if self.ag is not None:
            df = df.groupby(self.spatial_columns)[columns_to_agg].mean().reset_index()
        df.to_csv(self.paths["ag"], index=False)

    def create(self):
        self.df.to_csv(self.paths["complete"], index=False)
        self.aggregate()
        self.make_ml_ready()

        for ml, spl in CSVCollector.ml_split_combinations():
            s = Splitter(self.paths["ag"], split_strat=spl)
            train, test = s.split_it()

            train_key = CSVCollector.get_key_spatial(spl, "train", False)
            train.to_csv(self.paths[train_key], index=False)

            test_key = CSVCollector.get_key_spatial(spl, "test", False)
            test.to_csv(self.paths[test_key], index=False)

            train_key = CSVCollector.get_key_spatial(spl, "train", True)
            train = self.make_ml_ready_df(train)
            train.to_csv(self.paths[train_key], index=False)

            test_key = CSVCollector.get_key_spatial(spl, "test", True)
            test = self.make_ml_ready_df(test)
            test.to_csv(self.paths[test_key], index=False)

        return CSVCollector.collect(self.base_dir)

