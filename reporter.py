import numpy as np
import os
from datetime import datetime
import pandas as pd


class Reporter:
    def __init__(self, prefix, config_list, scenes_count, scenes_string, algorithms, repeat, folds):
        self.prefix = prefix
        self.config_list = config_list
        self.scenes_count = scenes_count
        self.scenes_string = scenes_string
        self.algorithms = algorithms
        self.repeat = repeat
        self.folds = folds
        self.details_columns = self.get_details_columns()
        self.summary_columns = self.get_summary_columns()
        self.details_text_columns = ["algorithm", "config"]
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"
        self.details = np.zeros((len(self.algorithms) * len(self.config_list), self.repeat * self.folds * 2))
        self.sync_details_file()
        self.create_log_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=self.details_text_columns, axis=1, inplace=True)
        self.details = df.to_numpy()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self, summary):
        df = pd.DataFrame(data=summary, columns=self.summary_columns)
        df.insert(0,"config",pd.Series([c["name"] for c in self.config_list]))
        df.insert(len(df.columns),"input",pd.Series(["-".join(c["input"]) for c in self.config_list]))
        df.insert(len(df.columns),"output",pd.Series([c["output"] for c in self.config_list]))
        df.insert(len(df.columns),"ag",pd.Series([(c["ag"] if c["ag"] is not None else "None") for c in self.config_list]))
        df.insert(len(df.columns),"scenes", pd.Series([c for c in self.scenes_count]))
        df.insert(len(df.columns),"scenes_string", pd.Series([c for c in self.scenes_string]))
        df.to_csv(self.summary_file, index=False)

    def find_mean_of_done_iterations(self, detail_cells):
        detail_cells = detail_cells[detail_cells != 0]
        if len(detail_cells) == 0:
            return 0
        else:
            return np.mean(detail_cells)

    def update_summary(self):
        score_mean = np.zeros((len(self.config_list), 2 * len(self.algorithms)))
        iterations = self.repeat * self.folds
        for index_config in range(len(self.config_list)):
            for index_algorithm in range(len(self.algorithms)):
                details_row = self.get_details_row(index_algorithm, index_config)
                detail_r2_cells = self.details[details_row, 0:iterations]
                r2_column_index = index_algorithm
                score_mean[index_config, r2_column_index] = np.mean(detail_r2_cells)
                detail_rmse_cells = self.details[details_row, iterations:]
                rmse_column_index = len(self.algorithms) + index_algorithm
                score_mean[index_config, rmse_column_index] = np.mean(detail_rmse_cells)
        self.write_summary(score_mean)

    def get_details_alg_conf(self):
        details_alg_conf = []
        for i in self.algorithms:
            for j in self.config_list:
                details_alg_conf.append((i,j["name"]))
        return details_alg_conf

    def get_details_row(self, index_algorithm, index_config):
        return index_algorithm*len(self.config_list) + index_config

    def get_details_column(self, repeat_number, fold_number, metric):
        #metric: 0,1: r2, rmse
        return (metric * self.repeat * self.folds ) + (repeat_number*self.folds + fold_number)

    def set_details(self, index_algorithm, repeat_number, fold_number, index_config, r2, rmse):
        details_row = self.get_details_row(index_algorithm, index_config)
        details_column_r2 = self.get_details_column(repeat_number, fold_number, 0)
        details_column_rmse = self.get_details_column(repeat_number, fold_number, 1)
        self.details[details_row, details_column_r2] = r2
        self.details[details_row, details_column_rmse] = rmse

    def get_details(self, index_algorithm, repeat_number, fold_number, index_config):
        details_row = self.get_details_row(index_algorithm, index_config)
        details_column_r2 = self.get_details_column(repeat_number, fold_number, 0)
        details_column_rmse = self.get_details_column(repeat_number, fold_number, 1)
        return self.details[details_row,details_column_r2], self.details[details_row,details_column_rmse]

    def get_details_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            for repeat in range(1,self.repeat+1):
                for fold in range(1,self.folds+1):
                    cols.append(f"{metric}({repeat}-{fold})")
        return cols

    def get_summary_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            for algorithm in self.algorithms:
                cols.append(f"{metric}({algorithm})")
        return cols

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.details_columns)
        details_alg_conf = self.get_details_alg_conf()
        algs = [i[0] for i in details_alg_conf]
        confs = [i[1] for i in details_alg_conf]

        df.insert(0,"algorithm",pd.Series(algs))
        df.insert(len(df.columns),"config",pd.Series(confs))

        df.to_csv(self.details_file, index=False)

    def log_scores(self, repeat_number, fold_number, algorithm, config, score, rmse):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{repeat_number} - {fold_number} - {algorithm} - {config}\n")
        log_file.write(str(score))
        log_file.write(str(rmse))
        log_file.write("\n")
        log_file.close()