from s2 import S2Extractor
from split_reporter import SplitReporter
from config_creator import ConfigCreator
from algorithm_runner import AlgorithmRunner
from split_ds_manager import SplitDSManager
from csv_collector import CSVCollector


class SplitEvaluator:
    def __init__(self, prefix="", verbose=False, algorithms=None, configs=None):
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "rf", "svr", "ann"]#, "ann", "cnn", "transformer", "lstm"]

        self.config_list = []
        self.trains = []
        self.tests = []
        self.scenes = []
        scenes_count = []
        scenes_string = []

        for config in configs:
            config_object = ConfigCreator.create_config_object(config)
            self.config_list.append(config_object)
            s2 = S2Extractor(ag=config_object["ag"], scenes=config_object["scenes"])
            paths, scenes = s2.process()
            train_key = CSVCollector.get_key_spatial(config_object["split_strat"], "train")
            self.trains.append(paths[train_key])
            test_key = CSVCollector.get_key_spatial(config_object["split_strat"], "test")
            self.tests.append(paths[test_key])
            self.scenes.append(scenes)
            scenes_count.append(len(scenes))
            scenes_string.append(scenes)

        self.reporter = SplitReporter(prefix, self.config_list, scenes_count, scenes_string, self.algorithms)

    def process(self):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(index_algorithm)

    def process_algorithm(self, index_algorithm):
        for index_config in range(len(self.config_list)):
            config = self.config_list[index_config]
            print("Start", f"{self.algorithms[index_algorithm]} - {config}")
            self.process_config(index_algorithm, index_config)

    def process_config(self, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        config = self.config_list[index_config]
        ds = SplitDSManager(train=self.trains[index_config], test=self.tests[index_config], x=config["input"], y=config["output"])
        train_x, train_y, test_x, test_y, validation_x, validation_y = ds.get_datasets()
        print("Train: ", self.trains[index_config])
        print("Test: ", self.tests[index_config])
        r2, rmse = self.reporter.get_details(index_algorithm, index_config)
        if r2 != 0:
            print(f"{index_algorithm}-{index_config} done already")
        else:
            r2, rmse = AlgorithmRunner.calculate_score(train_x, train_y, test_x, test_y, validation_x, validation_y, algorithm)
        if self.verbose:
            print(f"{r2} - {rmse}")
            print(f"R2 - RMSE")
        self.reporter.set_details(index_algorithm, index_config, r2, rmse)
        self.reporter.write_details()

