from translator import Translator


class ConfigCreator:
    @staticmethod
    def create_config_object(config):
        config_object = {"input":[],"output":"som","ag":"low","scenes":0,"name":None}
        if isinstance(config,str) or type(config) == list:
            config_object["input"] = Translator.get_columns_by_input_info(config)
        else:
            if isinstance(config["input"], str):
                config_object["input"] = Translator.get_columns_by_input_info(config["input"])
            else:
                config_object["input"] = config["input"]
            for a_prop in ["output","ag","scenes","name"]:
                if a_prop in config:
                    config_object[a_prop] = config[a_prop]

        if config_object["name"] is None:
            if isinstance(config, str):
                config_object["name"] = config
            elif isinstance(config["input"], str):
                config_object["name"] = config["input"]
            else:
                config_object["name"] = Translator.get_input_name(config)
            ag_name = "None"
            if config_object["ag"] is not None:
                ag_name = config_object["ag"]
            scene_part = config_object['scenes']
            if type(scene_part) == list:
                scene_part = len(config_object['scenes'])
            config_object["name"] = f"{config_object['name']}_{ag_name}_{scene_part}"

        return config_object
