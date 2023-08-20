from split_evaluator import SplitEvaluator
from splitter import Splitter


if __name__ == "__main__":
    base_config = {
        "ag": "low",
        "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
    }
    inputs = ["vis", "props_ex_som", "vis_props_ex_som", "bands", "all_ex_som"]
    configs = []
    for spl in Splitter.get_all_splits():
        for i in inputs:
            a_config = base_config.copy()
            a_config["input"] = i
            a_config["split"] = spl
            configs.append(a_config)
            break
        break

    train_path = "data/processed/47eb237b21511beb392f4845d460e399/train.csv"
    test_path = "data/processed/47eb237b21511beb392f4845d460e399/test.csv"
    c = SplitEvaluator(configs=configs, algorithms=["mlr","ann"], prefix="spl")
    c.process()
