import ds_manager
from s2 import S2Extractor
from reporter import Reporter
from config_creator import ConfigCreator
from algorithm_runner import AlgorithmRunner


class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, configs=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "plsr", "rf", "svr"]#, "ann", "cnn", "transformer", "lstm"]

        self.config_list = []
        self.csvs = []
        self.scenes = []
        scenes_count = []
        scenes_string = []

        for config in configs:
            config_object = ConfigCreator.create_config_object(config)
            self.config_list.append(config_object)
            s2 = S2Extractor(ag=config_object["ag"], scenes=config_object["scenes"])
            csv, scenes = s2.process()
            self.csvs.append(csv)
            self.scenes.append(scenes)
            scenes_count.append(len(scenes))
            scenes_string.append(scenes)

        self.reporter = Reporter(prefix, self.config_list, scenes_count, scenes_string,
                                 algorithms, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

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
            score = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if score != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                print("Start", f"{config}",f"{repeat_number}-{fold_number}")
                score = AlgorithmRunner.calculate_score(train_ds, test_ds, algorithm)
                self.reporter.log_scores(repeat_number, fold_number, algorithm, config, score)
            if self.verbose:
                print(f"{score}")
            self.reporter.set_details(index_algorithm, repeat_number, fold_number, index_config, score)
            self.reporter.write_details()
            self.reporter.update_summary()


if __name__ == "__main__":
    # configs = ["vis","props","vis-props","bands","upper-vis", "upper-vis-props","all_ex_som"]
    configs = ["vis","props_ex_som","vis_props_ex_som","bands","all_ex_som"]
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
