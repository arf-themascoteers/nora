from high_split_evaluator import HighSplitEvaluator
from hires1_splitter import Hires1Splitter


if __name__ == "__main__":
    base_config = {
    }

    configs = []
    a_config = base_config.copy()
    a_config["input"] = ["elevation", "moisture", "temp"]
    configs.append(a_config)

    c = HighSplitEvaluator(configs=configs, prefix="spl", algorithms=["mlr"])
    c.process()