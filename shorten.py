import pandas as pd


def shorten(orig, short):
    df = pd.read_csv(orig)
    df = df[df["som"]>1.72]
    df = df[df["som"]<3.29]
    df.to_csv(short,index=False)


if __name__ == "__main__":
    original = "data/vectis.csv"
    shorter = "data/shorter.csv"
    shorten(original, shorter)


