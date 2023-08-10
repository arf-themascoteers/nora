from sklearn import model_selection
import pandas as pd


class Splitter:
    def __init__(self, csv, mode="random"):
        #mode = random, spatial
        self.csv = csv
        self.mode = mode

    def split(self):
        df = pd.read_csv(self.csv)
        if self.mode == "random":
            return model_selection.train_test_split(df, test_size=0.2, random_state=2)
        df.sort_values(["row","column","scene"], inplace=True)
        total = len(df)
        test_count = int(total*.2)
        train_count = total - test_count
        train = df[0:train_count]
        test = df[train_count:]
        return train, test


if __name__ == "__main__":
    import reconstruct
    f = r"data/processed/47eb237b21511beb392f4845d460e399/ag.csv"
    s = Splitter(f, mode="spatial")
    train, test = s.split()
    reconstruct.recon(train)
    reconstruct.recon(test)
