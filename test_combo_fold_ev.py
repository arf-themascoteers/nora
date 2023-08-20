from combo_fold_evaluator import ComboFoldEvaluator


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
    }
    inputs = [
        ["B04", "B03", "B02"],
        ["red", "green", "blue"],
        ["B04", "B03", "B02", "red", "green", "blue"],
        ["B04", "B03", "B02", "elevation", "moisture", "temp"],
        ["red", "green", "blue", "elevation", "moisture", "temp"],
        ["B04", "B03", "B02", "red", "green", "blue", "elevation", "moisture", "temp"],
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "red", "green", "blue"],
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "red", "green", "blue","elevation", "moisture", "temp"],
    ]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = ComboFoldEvaluator(configs=configs, prefix="fold", folds=2)
    c.process()