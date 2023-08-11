from split_evaluator import SplitEvaluator


if __name__ == "__main__":
    configs = [
        {
            "input":"all_ex_som",
            "ag": "low",
            "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
        },
        {
            "input": "bands",
            "ag": "low",
            "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
        },
        {
            "input": "vis",
            "ag": "low",
            "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
        }
    ]
    train_path = "data/processed/47eb237b21511beb392f4845d460e399/train.csv"
    test_path = "data/processed/47eb237b21511beb392f4845d460e399/test.csv"
    c = SplitEvaluator(configs=configs, algorithms=["mlr","rf","ann"], prefix="top", train=train_path, test=test_path)
    c.process()