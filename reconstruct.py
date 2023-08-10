import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def recon(csv):
    df = None
    if isinstance(csv, str):
        df = pd.read_csv(f)
    else:
        df = csv
    df = df[df["scene"] == 1]
    min_row = df["row"].min()
    max_row = df["row"].max()
    min_col = df["column"].min()
    max_col = df["column"].max()
    print(min_row, max_row, min_col, max_col)
    min_row = int(min_row)
    max_row = int(max_row)
    min_col = int(min_col)
    max_col = int(max_col)
    x = np.zeros((max_row+1,max_col+1),dtype=np.float64)
    for i in range(len(df)):
        row = df.loc[i,"row"]
        col = df.loc[i,"column"]
        pix = df.loc[i,"B03"]
        row = int(row)
        col = int(col)
        x[row,col] = pix
    plt.imshow(x)
    plt.show()


if __name__ == "__main__":
    f = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    recon(f)