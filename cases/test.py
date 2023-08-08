from evaluator import Evaluator


if __name__ == "__main__":
    # configs = ["vis","props","vis-props","bands","upper-vis", "upper-vis-props","all_ex_som"]
    # configs = ["vis"]
    # configs = [
    #     {
    #         "input":["B02","B03"],
    #         "ag": "high",
    #         "scenes": ["S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040"],
    #         "name" : "shamsu"
    #     }
    # ]
    configs = ["vis","props_ex_som","vis_props_ex_som","bands","all_ex_som"]
    c = Evaluator(configs=configs, algorithms=["mlr","ann"],prefix="low",folds=3)
    c.process()