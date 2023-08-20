from single_fold_evaluator import SingleFoldEvaluator


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scene": "S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"
    }
    inputs = ["vis"]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = SingleFoldEvaluator(configs=configs, prefix="fold", folds=2, algorithms=["mlr"])
    c.process()