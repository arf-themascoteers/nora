class Translator:
    @staticmethod
    def get_input_name(config):
        inp = config["input"]
        if len(inp) == 1:
            return inp[0]
        return f"{inp[0]}+{len(inp)-1}"
    
    @staticmethod
    def get_bands():
        return ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    @staticmethod
    def get_columns_by_input_info(input_info):
        if input_info is None:
            input_info = "all_ex_som"

        if isinstance(input_info, str):
            if input_info == "vis":
                return Translator.get_vis_bands()
            elif input_info == "props":
                return Translator.get_soil_props()
            elif input_info == "props_ex_som":
                the_list = Translator.get_soil_props()
                the_list.remove("som")
                return the_list
            elif input_info.startswith("props_ex_prop_"):
                the_list = Translator.get_soil_props()
                the_prop = input_info[len("props_ex_prop_"):]
                the_list.remove(the_prop)
                return the_list
            elif input_info == "vis_props":
                return Translator.get_soil_props_vis()
            elif input_info == "vis_props_ex_som":
                the_list = Translator.get_soil_props_vis()
                the_list.remove("som")
                return the_list
            elif input_info == "upper_vis":
                return Translator.get_upper_vis_bands()
            elif input_info == "upper_vis_ex_props":
                the_list = Translator.get_upper_vis_bands()
                the_list.remove("som")
                return the_list
            elif input_info == "upper_vis_props":
                return Translator.get_soil_props_upper_vis()
            elif input_info == "upper_vis_props_ex_som":
                the_list = Translator.get_soil_props_upper_vis()
                the_list.remove("som")
                return the_list
            elif input_info == "bands":
                return Translator.get_bands()
            elif input_info == "all_ex_som":
                return Translator.get_all_ex_som()

        elif type(input_info) == list:
            return input_info

    @staticmethod
    def get_vis_bands():
        return ["B02", "B03", "B04"]

    @staticmethod
    def get_upper_vis_bands():
        return ["B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    @staticmethod
    def get_soil_props_upper_vis():
        return Translator.get_soil_props() + Translator.get_upper_vis_bands()

    @staticmethod
    def get_soil_props():
        return ["elevation", "moisture", "temp", "som"]

    @staticmethod
    def get_soil_props_vis():
        return Translator.get_soil_props() + Translator.get_vis_bands()

    @staticmethod
    def get_all_input():
        return Translator.get_bands() + Translator.get_soil_props()

    @staticmethod
    def get_superset():
        return Translator.get_all_input()

    @staticmethod
    def get_all_ex_som():
        superset = Translator.get_superset()
        superset.remove("som")
        return superset