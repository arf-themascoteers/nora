import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csv_collector import CSVCollector
from splitter import Splitter
import os


class Reconstructor:
    @staticmethod
    def recon(csv, height=None, width=None, save=True, pad=False):
        df = None
        if isinstance(csv, str):
            df = pd.read_csv(csv)
        else:
            df = csv
        if "scene" in df.columns:
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
            if pad:
                x[row-5:row+5,col-5:col+5] = pix
            else:
                x[row,col] = pix
        plt.imshow(x)
        file_name = os.path.basename(csv)
        if save:
            plt.savefig(f"plots/{file_name}.png")
        else:
            plt.show()
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
    #basedir = r"data/hi1p"
    path = CSVCollector.collect(basedir)
    height, width = Reconstructor.recon(path["ag"],pad=True)

    for s in Splitter.get_all_split_starts():
        train = path[CSVCollector.get_key_spatial(s,"train", ml_ready=False)]
        test = path[CSVCollector.get_key_spatial(s,"test", ml_ready=False)]
        Reconstructor.recon(train, height, width,pad=False)
        Reconstructor.recon(test, height, width,pad=False)
