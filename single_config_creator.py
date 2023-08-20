from translator import Translator


class SingleConfigCreator:
    @staticmethod
    def create_config_object(config):
        config_object = {"input":[],"output":"som",
                         "split_strat":"random",
                         "scene":"S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511","name":None}

        if isinstance(config["input"], str):
            config_object["input"] = Translator.get_columns_by_input_info(config["input"])
        else:
            config_object["input"] = config["input"]

        for a_prop in ["output","ag","split_strat","scenes","name"]:
            if a_prop in config:
                config_object[a_prop] = config[a_prop]

        if config_object["name"] is None:
            if isinstance(config["input"], str):
                config_object["name"] = config["input"]
            else:
                config_object["name"] = Translator.get_input_name(config)

            config_object["name"] = f"{config_object['name']}_{config_object['split_strat']}"

        return config_object
