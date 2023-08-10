from evaluator import Evaluator
from translator import Translator


if __name__ == "__main__":
    configs = []#["vis","props","vis-props","bands","upper-vis", "upper-vis-props","all_ex_som"]
    base_config = {
            "input":[],
            "ag": "low",
            "scenes": ["S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040"]
        }
    for prop in Translator.get_superset():
        if prop == "som":
            continue
        a_config = base_config.copy()
        a_config["input"] = [prop]
        configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_vis_bands() + ["elevation"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_vis_bands() + ["temp"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_vis_bands() + ["moisture"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_bands() + ["elevation"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_bands() + ["temp"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_bands() + ["moisture"]
    configs.append(a_config)

    a_config = base_config.copy()
    a_config["input"] = Translator.get_vis_bands() + ["B08","B11"]
    configs.append(a_config)

    c = Evaluator(configs=configs, algorithms=["mlr","ann"],prefix="many",folds=3, repeat=1)
    c.process()