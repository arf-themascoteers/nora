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
            self.algorithms = ["mlr", "plsr", "rf", "svr"]#, "ann", "cnn", "transformer", "lstm"]

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
            self.trains.append(paths[CSVCollector.get_key_spatial(config_object["split_strat"], "train")])
            self.tests.append(paths[CSVCollector.get_key_spatial(config_object["split_strat"], "test")])
            self.scenes.append(scenes)
            scenes_count.append(len(scenes))
            scenes_string.append(scenes)

        self.reporter = SplitReporter(prefix, self.config_list, scenes_count, scenes_string, algorithms)

    def process(self):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(index_algorithm)

    def process_algorithm(self, index_algorithm):
        print("Start", f"{self.algorithms[index_algorithm]}")
        for index_config in range(len(self.config_list)):
            config = self.config_list[index_config]
            print("Start", f"{config}")
            self.process_config(index_algorithm, index_config)

    def process_config(self, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        config = self.config_list[index_config]
        ds = SplitDSManager(train=self.trains[index_config],test=self.tests[index_config], x=config["input"], y=config["output"])
        train_ds, test_ds = ds.get_datasets()
        r2, rmse = self.reporter.get_details(index_algorithm, index_config)
        if r2 != 0:
            print(f"{index_algorithm}-{index_config} done already")
        else:
            print("Start", f"{config}",f"{index_algorithm}-{index_config}")
            r2, rmse = AlgorithmRunner.calculate_score(train_ds, test_ds, algorithm)
        if self.verbose:
            print(f"{r2} - {rmse}")
            print(f"R2 - RMSE")
        self.reporter.set_details(index_algorithm, index_config, r2, rmse)
        self.reporter.write_details()

