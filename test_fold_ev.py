from fold_evaluator import FoldEvaluator


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
    }
    inputs = [
                ["B02", "B03", "B04", "B08"],
                ["B02", "B03", "B04", "B08", "elevation", "temp", "moisture"]
              ]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = FoldEvaluator(configs=configs, prefix="fold", folds=10)
    c.process()