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

        s = Splitter(self.paths["ag"], mode="random")
        train, test = s.split_it()
        train.to_csv(self.paths[CSVCollector.get_key_random("train")], index=False)
        test.to_csv(self.paths[CSVCollector.get_key_random("test")], index=False)

        for spl in Splitter.get_all_split_starts():
            s = Splitter(self.paths["ag"], mode="spatial", split_strat=spl)
            train, test = s.split_it()
            train.to_csv(self.paths[f"train_{spl}"], index=False)
            test.to_csv(self.paths[f"test_{spl}"], index=False)

        return CSVCollector.collect(self.base_dir)
