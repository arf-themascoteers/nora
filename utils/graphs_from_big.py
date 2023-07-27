import matplotlib.pyplot as plt
import os
import pandas as pd
import ds_manager
from torch.utils.data import DataLoader
os.chdir("../")


d = ds_manager.DSManager(config="bands")


for fold_number, (train_ds, test_ds) in enumerate(d.get_k_folds()):
    dataloader = DataLoader(train_ds, batch_size=3, shuffle=True)
    for (x, y) in dataloader:
        print(x.shape)
        print(y.shape)
        x = x[0]
        plt.scatter(list(range(x.shape[0])),x)
        plt.xlabel("Band Index")
        plt.ylabel("Normalized value")
        for i in range(x.shape[0]):
            plt.text(i,x[i],f"{i}")
        #plt.plot(x)
        plt.show()
        exit(0)
