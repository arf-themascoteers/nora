import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from splitter import Splitter
from csv_collector import CSVCollector


class CSVCreator:
    def __init__(self, df, export_dir, ag):
        self.df = df
        self.ag = ag
        self.complete_csv_path, self.ag_csv_path, self.ml_csv_path, self.train_csv_path, self.test_csv_path = CSVCollector.collect(export_dir)
        self.geo_columns = ["lon", "lat", "when"]
        self.spatial_columns = ["scene", "row", "column"]

    def make_ml_ready(self):
        df = pd.read_csv(self.ag_csv_path)
        df = self.make_ml_ready_df(df)
        df.to_csv(self.ml_csv_path, index=False)

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
        df = pd.read_csv(self.complete_csv_path)
        df.drop(columns=self.geo_columns, axis=1, inplace=True)
        columns_to_agg = df.columns.drop(self.spatial_columns)
        if self.ag is not None:
            df = df.groupby(self.spatial_columns)[columns_to_agg].mean().reset_index()
        df.to_csv(self.ag_csv_path, index=False)

    def create(self):
        self.df.to_csv(self.complete_csv_path, index=False)
        self.aggregate()
        s = Splitter(self.ag_csv_path, mode="spatial")
        train, test = s.split()
        df = pd.concat([train, test])
        df = self.make_ml_ready_df(df)
        train = df[0:len(train)]
        test = df[len(train):]
        train.to_csv(self.train_csv_path, index=False)
        test.to_csv(self.test_csv_path, index=False)
        self.make_ml_ready()
        return self.complete_csv_path, self.ag_csv_path, self.ml_csv_path, self.train_csv_path, self.test_csv_path
