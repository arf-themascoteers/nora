from fold_evaluator import FoldEvaluator


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
    }
    inputs = ["vis", "props_ex_som", "vis_props_ex_som", "bands", "all_ex_som"]
    configs = []

    for i in inputs:
        a_config = base_config.copy()
        a_config["input"] = i
        configs.append(a_config)

    c = FoldEvaluator(configs=configs, algorithms=["mlr","ann"], prefix="fold")
    c.process()