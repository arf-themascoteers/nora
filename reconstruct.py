import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csv_collector import CSVCollector
from splitter import Splitter
from sklearn import model_selection
import os


class Reconstructor:
    @staticmethod
    def recon(csv, height=None, width=None):
        df = None
        if isinstance(csv, str):
            df = pd.read_csv(csv)
        else:
            df = csv
        df = df[df["scene"] == 1]
        if height is None or width is None:
            max_row = df["row"].max()
            max_col = df["column"].max()
            height = int(max_row)
            width = int(max_col)
        x = np.zeros((height+1,width+1),dtype=np.float64)
        for i in df.index:
            row = df.loc[i,"row"]
            col = df.loc[i,"column"]
            pix = df.loc[i,"B03"]
            row = int(row)
            col = int(col)
            x[row,col] = pix
        plt.imshow(x)
        file_name = os.path.basename(csv)
        plt.savefig(f"plots/{file_name}.png")
        plt.clf()
        return height, width

    @staticmethod
    def recon_folder(folder):
        paths = CSVCollector.collect(folder)
        height, width = Reconstructor.recon(paths["ag"])
        Reconstructor.recon(paths["train_spatial_csv_path"], height, width)
        Reconstructor.recon(paths["test_spatial_csv_path"], height, width)


if __name__ == "__main__":
    basedir = r"data/processed/47eb237b21511beb392f4845d460e399"
    path = CSVCollector.collect(basedir)
    height, width = Reconstructor.recon(path["ag"])

    train = path[CSVCollector.get_key_random("train")]
    test = path[CSVCollector.get_key_random("test")]
    Reconstructor.recon(train, height, width)
    Reconstructor.recon(test, height, width)

    for s in Splitter.get_all_split_starts():
        train = path[CSVCollector.get_key_spatial(s,"train")]
        test = path[CSVCollector.get_key_spatial(s,"test")]
        Reconstructor.recon(train, height, width)
        Reconstructor.recon(test, height, width)
