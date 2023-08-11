from sklearn import model_selection
import pandas as pd


class Splitter:
    def __init__(self, csv, mode="random",strat="mid"):
        #mode = random, spatial
        #strat = top, bottom, right, left, mid
        self.csv = csv
        self.mode = mode
        self.strat = strat

    def split(self):
        df = pd.read_csv(self.csv)
        if self.mode == "random":
            return model_selection.train_test_split(df, test_size=0.2, random_state=2)
        df.sort_values(["row","column","scene"], inplace=True)
        total = len(df)
        test_portion = 0.2
        test_count = int(total*.2)
        train_portion = 1 - test_portion
        train_count = total - test_count
        train, test = None, None

        if self.strat == "mid":
            train_portion_each_block = train_portion/2
            train_count_first_block = int(total * train_portion_each_block)

            train_first = df[0:train_count_first_block]
            test = df[train_count_first_block:train_count_first_block+test_count]
            train_second = df[train_count_first_block+test_count:]
            train = pd.concat([train_first, train_second])

        elif self.strat == "top":
            train = df[0: train_count]
            test = df[train_count:]

        elif self.strat == "bottom":
            test = df[0:test_count]
            train = df[test_count: ]

        return train, test



