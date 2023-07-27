import numpy as np
import pandas as pd
import ds_manager
import os
from datetime import datetime
import torch
from ann import ANN
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, configs=None):
        self.configs = configs
        if self.configs is None:
            self.configs = []
        self.TEST = False
        self.TEST_SCORE = 0
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "plsr", "rf", "svr"]#, "ann", "cnn", "transformer", "lstm"]

        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"

        self.details = np.zeros(( len(self.algorithms) * len(self.configs), self.repeat*self.folds))
        self.details_index = self.get_details_index()

        self.sync_details_file()
        self.create_log_file()

    def get_config_name(self, config):
        if isinstance(config, str):
            return config
        if len(config) == 1:
            return config[0]
        return config[0]+"+"

    def get_details_index(self):
        details_index = []
        for i in range(len(self.algorithms)):
            for j in range(len(self.configs)):
                details_index.append(f"I-{self.algorithms[i]}-{self.configs[j]}")
        return details_index

    def get_details_columns(self):
        details_columns = []
        for i in range(self.repeat):
            for j in range(self.folds):
                details_columns.append(f"I-{i}-{j}")
        return details_columns

    def get_details_row(self, index_algorithm, index_config):
        return index_algorithm*len(self.configs) + index_config

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

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.details = df.to_numpy()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self, summary):
        df = pd.DataFrame(data=summary, columns=self.algorithms, index=self.get_summary_index())
        df.to_csv(self.summary_file)

    def get_summary_index(self):
        return [self.get_config_name(config) for config in self.configs]

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.get_details_columns(), index=self.get_details_index())
        df.to_csv(self.details_file)

    def log_scores(self, repeat_number, fold_number, algorithm, config, score):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{repeat_number} - {fold_number} - {algorithm} - {config}\n")
        log_file.write(str(score))
        log_file.write("\n")
        log_file.close()

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def update_summary(self):
        score_mean = np.zeros((len(self.configs),len(self.algorithms)))
        for index_config in range(len(self.configs)):
            for index_algorithm in range(len(self.algorithms)):
                details_row = self.get_details_row(index_algorithm, index_config)
                detail_cells = self.details[details_row, :]
                detail_cells = detail_cells[detail_cells != 0]
                if len(detail_cells) == 0:
                    score_mean[index_config, index_algorithm] = 0
                else:
                    score_mean[index_config, index_algorithm] = np.mean(detail_cells)
        self.write_summary(score_mean)

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]}")
        for index_config in range(len(self.configs)):
            config = self.configs[index_config]
            print("Start", f"{config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        config = self.configs[index_config]
        ds = ds_manager.DSManager(folds=self.folds, config=config)
        for fold_number, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
            score = self.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if score != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                print("Start", f"{config}")
                score = self.calculate_score(train_ds, test_ds, algorithm)
                self.log_scores(repeat_number, fold_number, algorithm, config, score)
            if self.verbose:
                print(f"{score}")
            self.set_details(index_algorithm, repeat_number, fold_number, index_config, score)
            self.write_details()
            self.update_summary()

    def calculate_score(self, train_ds, test_ds, algorithm):
        if self.TEST:
            self.TEST_SCORE = self.TEST_SCORE + 1
            return self.TEST_SCORE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if algorithm == "ann":
            model = ANN(device, train_ds, test_ds)
            model.train_model()
            return model.test()
        else:
            train_x = train_ds.x
            train_y = train_ds.y
            test_x = test_ds.x
            test_y = test_ds.y

            model_instance = None
            if algorithm == "mlr":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()

            model_instance = model_instance.fit(train_x, train_y)
            return model_instance.score(test_x, test_y)

    def create_summary_index(self):
        index = []
        for alg in self.algorithms:
            index.append(alg)
        return index


if __name__ == "__main__":
    configs = ["vis","props","vis-props","bands","all"]
    c = Evaluator(configs=configs, algorithms=["mlr","ann"],prefix="both",folds=3)
    c.process()
