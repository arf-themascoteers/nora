import os
from splitter import Splitter


class CSVCollector:
    @staticmethod
    def collect(source_dir):
        train_random = CSVCollector.get_key_random("train")
        test_random = CSVCollector.get_key_random("test")
        path = {
            "complete": os.path.join(source_dir, "complete.csv"),
            "ag": os.path.join(source_dir, "ag.csv"),
            "ml": os.path.join(source_dir, "ml.csv"),
            train_random: os.path.join(source_dir, f"{train_random}.csv"),
            test_random: os.path.join(source_dir, f"{test_random}.csv")
        }
        for spl in Splitter.get_all_split_starts():
            train = CSVCollector.get_key_spatial(spl, "train")
            test = CSVCollector.get_key_spatial(spl, "test")
            path[train] = os.path.join(source_dir, f"{train}.csv")
            path[test] = os.path.join(source_dir, f"{test}.csv")
        return path

    @staticmethod
    def get_key_spatial(split_strat, task):
        if split_strat is None:
            return CSVCollector.get_key_random(task)
        return f"{task}_{split_strat}"

    @staticmethod
    def get_key_random(task):
        return f"{task}_random"

        