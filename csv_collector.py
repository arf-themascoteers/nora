import os
from splitter import Splitter


class CSVCollector:
    @staticmethod
    def collect(source_dir):
        path = {
            "complete": os.path.join(source_dir, "complete.csv"),
            "ag": os.path.join(source_dir, "ag.csv"),
            "ml": os.path.join(source_dir, "ml.csv"),
        }
        for ml, spl, task in CSVCollector.ml_split_task_combinations():
            key = CSVCollector.get_key_spatial(spl, task, ml_ready=ml)
            path[key] = os.path.join(source_dir, f"{key}.csv")
        return path

    @staticmethod
    def get_key_spatial(split_strat, task, ml_ready=True):
        key = f"{task}_{split_strat}"
        if ml_ready:
            key = f"{key}_ml"
        return key

    @staticmethod
    def ml_split_combinations():
        for ml in [True, False]:
                for spl in Splitter.get_all_split_starts():
                    yield ml, spl

    @staticmethod
    def ml_split_task_combinations():
        for ml, spl in CSVCollector.ml_split_combinations():
            for task in ["train","test"]:
                    yield ml, spl, task

