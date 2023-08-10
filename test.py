from evaluator import Evaluator


if __name__ == "__main__":
    configs = [
        {
            "input":"props_ex_som",
            "ag": "low",
            "scenes": ["S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511"]
        }
    ]
    c = Evaluator(configs=configs, algorithms=["mlr","ann"],prefix="test",folds=2)
    c.process()