from high_fold_evaluator import HighFoldEvaluator


if __name__ == "__main__":
    base_config = {
    }
    inputs = [
        ["B04","B03","B02"],
        ["B04","B03","B02","B08"],
        ["B08"],
        ["elevation", "moisture", "temp"],
        ["elevation", "moisture", "temp","B04","B03","B02"],
        ["elevation", "moisture", "temp","B04","B03","B02","B08"],
        ["elevation", "moisture", "temp","B08"]
    ]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = HighFoldEvaluator(configs=configs, prefix="high_fold", folds=2, algorithms=["ann"])
    c.process()