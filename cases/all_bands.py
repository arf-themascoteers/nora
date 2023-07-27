import os
from evaluator import Evaluator
from ds_manager import DSManager

os.chdir("../")

bands = DSManager.get_bands()
configs = [[band] for band in bands]
c = Evaluator(configs=configs, algorithms=["ann"], prefix="all_bands")
c.process()

