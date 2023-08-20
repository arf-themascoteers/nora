from split_evaluator import SplitEvaluator
from splitter import Splitter


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
    }
    inputs = ["elevation", "moisture", "temp"]
    configs = []

    a_config = base_config.copy()
    a_config["input"] = inputs
    a_config["split_strat"] = "random"
    configs.append(a_config)

    c = SplitEvaluator(configs=configs, prefix="spl", algorithms=["mlr"])
    c.process()
