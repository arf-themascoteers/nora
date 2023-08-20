from high_fold_evaluator import HighFoldEvaluator


if __name__ == "__main__":
    base_config = {
    }
    inputs = [
        ["elevation", "moisture", "temp"],
    ]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = HighFoldEvaluator(configs=configs, prefix="high_fold_2", folds=10, algorithms=["mlr"])
    c.process()