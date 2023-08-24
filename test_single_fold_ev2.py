from single_fold_evaluator import SingleFoldEvaluator
from base_path import BASE_PATH
import os


if __name__ == "__main__":
    inputs = ["vis"]
    configs = []

    for i in os.listdir(BASE_PATH):
        path = os.path.join(BASE_PATH, i)
        if not os.path.isdir(path):
            continue
        if not i.startswith("S2"):
            continue
        a_config = {"input": "vis", "scene":i}
        configs.append(a_config)

    print(configs)
    c = SingleFoldEvaluator(configs=configs, prefix="5_Scenes", folds=2, algorithms=["mlr"])
    c.process()