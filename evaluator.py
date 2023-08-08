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
from s2 import S2Extractor


class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, configs=None):
        self.TEST = False
        self.TEST_SCORE = 0
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "plsr", "rf", "svr"]#, "ann", "cnn", "transformer", "lstm"]

        self.details_text_columns = ["algorithm", "config"]

        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"

        self.details = np.zeros(( len(self.algorithms) * len(configs), self.repeat*self.folds))

        self.config_list = []
        for config in configs:
            self.config_list.append(Evaluator.create_config_object(config))

        self.sync_details_file()
        self.create_log_file()

        self.scenes = []
        self.csvs = []

    @staticmethod
    def get_input_name(config):
        inp = config["input"]
        if len(inp) == 1:
            return inp[0]
        return f"{inp[0]}+{len(inp)-1}"

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
        df.insert(len(df.columns),"scenes",pd.Series([len(c) for c in self.scenes]))
        df.insert(len(df.columns),"scenes_string",pd.Series([S2Extractor.create_scenes_string(c) for c in self.scenes]))
        df.to_csv(self.summary_file, index=False)

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

    def process(self):
        self.extract()
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

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

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]}")
        for index_config in range(len(self.config_list)):
            config = self.config_list[index_config]
            print("Start", f"{config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        config = self.config_list[index_config]
        ds = ds_manager.DSManager(self.csvs[index_config], folds=self.folds, x=config["input"], y=config["output"])
        for fold_number, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
            score = self.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if score != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                print("Start", f"{config}",f"{repeat_number}-{fold_number}")
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

    @staticmethod
    def create_config_object(config):
        config_object = {"input":[],"output":"som","ag":"low","scenes":0,"name":None}
        if isinstance(config,str) or type(config) == list:
            config_object["input"] = Evaluator.get_columns_by_input_info(config)
        else:
            if isinstance(config["input"], str):
                config_object["input"] = Evaluator.get_columns_by_input_info(config["input"])
            else:
                config_object["input"] = config["input"]
            for a_prop in ["output","ag","scenes","name"]:
                if a_prop in config:
                    config_object[a_prop] = config[a_prop]

        if config_object["name"] is None:
            if isinstance(config, str):
                config_object["name"] = config
            else:
                config_object["name"] = Evaluator.get_input_name(config)
            ag_name = "None"
            if config_object["ag"] is not None:
                ag_name = config_object["ag"]
            config_object["name"] = f"{config_object['name']}_{ag_name}_{config_object['scenes']}"

        return config_object

    @staticmethod
    def get_bands():
        return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    @staticmethod
    def get_columns_by_input_info(input_info):
        if input_info is None:
            input_info = "all"

        if isinstance(input_info, str):
            if input_info == "vis":
                return Evaluator.get_vis_bands()
            elif input_info == "props":
                return Evaluator.get_soil_props()
            elif input_info == "props_ex_som":
                the_list = Evaluator.get_soil_props()
                the_list.remove("som")
                return the_list
            elif input_info.startswith("props_ex_prop_"):
                the_list = Evaluator.get_soil_props()
                the_prop = input_info[len("props_ex_prop_"):]
                the_list.remove(the_prop)
                return the_list
            elif input_info == "vis_props":
                return Evaluator.get_soil_props_vis()
            elif input_info == "vis_props_ex_som":
                the_list = Evaluator.get_soil_props_vis()
                the_list.remove("som")
                return the_list
            elif input_info == "upper_vis":
                return Evaluator.get_upper_vis_bands()
            elif input_info == "upper_vis_ex_props":
                the_list = Evaluator.get_upper_vis_bands()
                the_list.remove("som")
                return the_list
            elif input_info == "upper_vis_props":
                return Evaluator.get_soil_props_upper_vis()
            elif input_info == "upper_vis_props_ex_som":
                the_list = Evaluator.get_soil_props_upper_vis()
                the_list.remove("som")
                return the_list
            elif input_info == "bands":
                return Evaluator.get_bands()
            elif input_info == "all":
                return Evaluator.get_sperset()

        elif type(input_info) == list:
            return input_info

    @staticmethod
    def get_vis_bands():
        return ["B02", "B03", "B04"]

    @staticmethod
    def get_upper_vis_bands():
        return ["B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    @staticmethod
    def get_soil_props_upper_vis():
        return Evaluator.get_soil_props() + Evaluator.get_upper_vis_bands()

    @staticmethod
    def get_soil_props():
        return ["elevation", "moisture", "temp", "som"]

    @staticmethod
    def get_soil_props_vis():
        return Evaluator.get_soil_props() + Evaluator.get_vis_bands()

    @staticmethod
    def get_all_input():
        return Evaluator.get_bands() + Evaluator.get_soil_props()

    @staticmethod
    def get_sperset():
        return Evaluator.get_all_input()

    @staticmethod
    def get_all_except_output(the_output):
        superset = Evaluator.get_sperset()
        superset.remove(the_output)
        return superset

    def get_details_columns(self):
        cols = []
        for repeat in range(1,self.repeat+1):
            for fold in range(1,self.folds+1):
                cols.append(f"I-{repeat}-{fold}")
        return cols

    def extract(self):
        for config in self.config_list:
            s2 = S2Extractor(ag=config["ag"], scenes=config["scenes"])
            csv, scenes = s2.process()
            self.csvs.append(csv)
            self.scenes.append(scenes)


if __name__ == "__main__":
    # configs = ["vis","props","vis-props","bands","upper-vis", "upper-vis-props","all"]
    configs = ["vis","props","vis_props","bands","all"]
    # configs = ["vis"]
    # configs = [
    #     {
    #         "input":["B02","B03"],
    #         "ag": "high",
    #         "scenes": ["S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040"],
    #         "name" : "shamsu"
    #     }
    # ]

    c = Evaluator(configs=configs, algorithms=["mlr","ann"],prefix="both",folds=3)
    c.process()
