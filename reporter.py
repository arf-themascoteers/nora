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
        self.details_text_columns = ["algorithm", "config"]
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"
        self.details = np.zeros((len(self.algorithms) * len(self.config_list), self.repeat * self.folds))
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
        df = pd.DataFrame(data=summary, columns=self.algorithms)
        df.insert(0,"config",pd.Series([c["name"] for c in self.config_list]))
        df.insert(len(df.columns),"input",pd.Series(["-".join(c["input"]) for c in self.config_list]))
        df.insert(len(df.columns),"output",pd.Series([c["output"] for c in self.config_list]))
        df.insert(len(df.columns),"ag",pd.Series([c["ag"] for c in self.config_list]))
        df.insert(len(df.columns),"scenes", pd.Series([c for c in self.scenes_count]))
        df.insert(len(df.columns),"scenes_string", pd.Series([c for c in self.scenes_string]))
        df.to_csv(self.summary_file, index=False)

    def update_summary(self):
        score_mean = np.zeros((len(self.config_list),len(self.algorithms)))
        for index_config in range(len(self.config_list)):
            for index_algorithm in range(len(self.algorithms)):
                details_row = self.get_details_row(index_algorithm, index_config)
                detail_cells = self.details[details_row, :]
                detail_cells = detail_cells[detail_cells != 0]
                if len(detail_cells) == 0:
                    score_mean[index_config, index_algorithm] = 0
                else:
                    score_mean[index_config, index_algorithm] = np.mean(detail_cells)
        self.write_summary(score_mean)

    def get_details_alg_conf(self):
        details_alg_conf = []
        for i in self.algorithms:
            for j in self.config_list:
                details_alg_conf.append((i,j["name"]))
        return details_alg_conf

    def get_details_row(self, index_algorithm, index_config):
        return index_algorithm*len(self.config_list) + index_config

    def get_details_column(self, repeat_number, fold_number):
        return repeat_number*self.folds + fold_number

    def set_details(self, index_algorithm, repeat_number, fold_number, index_config, score):
        details_row = self.get_details_row(index_algorithm, index_config)
        details_column = self.get_details_column(repeat_number, fold_number)
        self.details[details_row,details_column] = score

    def get_details(self, index_algorithm, repeat_number, fold_number, index_config):
        details_row = self.get_details_row(index_algorithm, index_config)
        details_column = self.get_details_column(repeat_number, fold_number)
        return self.details[details_row,details_column]

    def get_details_columns(self):
        cols = []
        for repeat in range(1,self.repeat+1):
            for fold in range(1,self.folds+1):
                cols.append(f"I-{repeat}-{fold}")
        return cols

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.get_details_columns())
        details_alg_conf = self.get_details_alg_conf()
        algs = [i[0] for i in details_alg_conf]
        confs = [i[1] for i in details_alg_conf]

        df.insert(0,"algorithm",pd.Series(algs))
        df.insert(len(df.columns),"config",pd.Series(confs))

        df.to_csv(self.details_file, index=False)

    def log_scores(self, repeat_number, fold_number, algorithm, config, score):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{repeat_number} - {fold_number} - {algorithm} - {config}\n")
        log_file.write(str(score))
        log_file.write("\n")
        log_file.close()