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
        self.details_text_columns = ["algorithm", "config"]
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((len(self.algorithms) * len(self.config_list), 2))
        self.sync_details_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=self.details_text_columns, axis=1, inplace=True)
        self.details = df.to_numpy()

    def get_details_alg_conf(self):
        details_alg_conf = []
        for i in self.algorithms:
            for j in self.config_list:
                details_alg_conf.append((i,j["name"]))
        return details_alg_conf

    def get_details_row(self, index_algorithm, index_config):
        return index_algorithm*len(self.config_list) + index_config

    def set_details(self, index_algorithm, index_config, r2, rmse):
        details_row = self.get_details_row(index_algorithm, index_config)
        self.details[details_row, 0] = r2
        self.details[details_row, 1] = rmse

    def get_details(self, index_algorithm, index_config):
        details_row = self.get_details_row(index_algorithm, index_config)
        return self.details[details_row,0], self.details[details_row,1]

    def get_details_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            cols.append(f"{metric}")
        return cols

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        details_alg_conf = self.get_details_alg_conf()
        algs = [i[0] for i in details_alg_conf]
        confs = [i[1] for i in details_alg_conf]

        df.insert(0,"algorithm",pd.Series(algs))
        df.insert(len(df.columns),"config",pd.Series(confs))

        df.to_csv(self.details_file, index=False)
