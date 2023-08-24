from single_fold_evaluator import SingleFoldEvaluator
from base_path import BASE_PATH
import os


if __name__ == "__main__":
    inputs = ["props_ex_som", "bands", "all_ex_som"]
    configs = []

    for i in inputs:
        for scene in os.listdir(BASE_PATH):
            path = os.path.join(BASE_PATH, scene)
            if not os.path.isdir(path):
                continue
            if not scene.startswith("S2"):
                continue
            a_config = {"input": i, "scene": scene}
            configs.append(a_config)

    c = SingleFoldEvaluator(configs=configs, prefix="5_Scenes_all", folds=2)
    c.process()