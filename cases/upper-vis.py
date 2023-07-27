import os
from evaluator import Evaluator
from ds_manager import DSManager

os.chdir("../")

bands = DSManager.get_all()
configs = ["upper-vis", "upper-vis-props"]
c = Evaluator(configs=configs, algorithms=["mlr","ann"], prefix="upper-vis")
c.process()

