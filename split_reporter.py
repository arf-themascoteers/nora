import numpy as np
import os
from datetime import datetime
import pandas as pd


class SplitReporter:
    def __init__(self, prefix, config_list, scenes_count, scenes_string, algorithms):
        self.prefix = prefix
        self.config_list = config_list
        self.scenes_count = scenes_count
        self.scenes_string = scenes_string
        self.algorithms = algorithms
        self.details_columns = self.get_details_columns()
        self.details_text_columns = ["config"]
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((len(self.config_list),len(self.algorithms) * 2))
        self.sync_details_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=self.details_text_columns, axis=1, inplace=True)
        self.details = df.to_numpy()

    def get_details_column(self, index_algorithm, metric):
        return (metric * len(self.algorithms) ) + index_algorithm

    def set_details(self, index_algorithm, index_config, r2, rmse):
        r2_column = self.get_details_column(index_algorithm, 0)
        self.details[index_config, r2_column] = r2
        rmse_column = self.get_details_column(index_algorithm, 1)
        self.details[index_config, rmse_column] = rmse

    def get_details(self, index_algorithm, index_config):
        r2_column = self.get_details_column(index_algorithm, 0)
        rmse_column = self.get_details_column(index_algorithm, 1)
        return self.details[index_config, r2_column], self.details[index_config, rmse_column]

    def get_details_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            for algorithm in self.algorithms:
                cols.append(f"{metric}({algorithm})")
        return cols

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        df.insert(0,"config",pd.Series([c["name"] for c in self.config_list]))
        df.to_csv(self.details_file, index=False)
