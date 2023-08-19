import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csv_collector import CSVCollector
from splitter import Splitter
from sklearn import model_selection


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
        plt.show()
        return height, width

    @staticmethod
    def recon_folder(folder):
        paths = CSVCollector.collect(folder)
        height, width = Reconstructor.recon(paths["ag_csv_path"])
        Reconstructor.recon(paths["train_spatial_csv_path"], height, width)
        Reconstructor.recon(paths["test_spatial_csv_path"], height, width)


if __name__ == "__main__":
    basedir = r"data/processed/47eb237b21511beb392f4845d460e399"
    f1 = r"data/ag.csv"
    # f2 = r"data/processed/47eb237b21511beb392f4845d460e399/train_spatial.csv"
    # f3 = r"data/processed/47eb237b21511beb392f4845d460e399/test_spatial.csv"
    # height, width = Reconstructor.recon(f1)
    # Reconstructor.recon(f2, height, width)
    # Reconstructor.recon(f3, height, width)

    height, width = Reconstructor.recon(f1)
    #for s in ["top", "bottom", "mid", "left", "right", "block"]:
    for s in ["block"]:
        df = pd.read_csv(f1)
        train, test = model_selection.train_test_split(df, test_size=0.2, random_state=2)
        Reconstructor.recon(train, height, width)
        Reconstructor.recon(test, height, width)
