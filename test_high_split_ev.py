from high_split_evaluator import HighSplitEvaluator
from hires1_splitter import Hires1Splitter


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

    for spl in Hires1Splitter.get_all_split_starts():
        for i in inputs:
            a_config = base_config.copy()
            a_config["input"] = i
            configs.append(a_config)

    c = HighSplitEvaluator(configs=configs, prefix="spl")
    c.process()