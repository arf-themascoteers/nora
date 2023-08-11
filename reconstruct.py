import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    for i in range(len(df)):
        row = df.loc[i,"row"]
        col = df.loc[i,"column"]
        pix = df.loc[i,"B03"]
        row = int(row)
        col = int(col)
        x[row,col] = pix
    plt.imshow(x)
    plt.show()
    return height, width


if __name__ == "__main__":
    f1 = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    f2 = r"data/processed/47eb237b21511beb392f4845d460e399/train_spatial.csv"
    f3 = r"data/processed/47eb237b21511beb392f4845d460e399/test_spatial.csv"
    height, width = recon(f1)
    recon(f2, height, width)
    recon(f3, height, width)