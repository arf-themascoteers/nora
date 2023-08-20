from sklearn import model_selection
import pandas as pd
import random


class Splitter:
    def __init__(self, csv, split_strat="random"):
        self.csv = csv
        self.split_strat = split_strat

    @staticmethod
    def get_all_split_starts():
        return ["random", "top", "bottom", "mid", "left", "right","block"]

    def split_it(self):
        df = pd.read_csv(self.csv)
        if self.split_strat == "random":
            return model_selection.train_test_split(df, test_size=0.1, random_state=2)

        total = len(df)
        test_portion = 0.2
        test_count = int(total*.2)
        train_portion = 1 - test_portion
        train_count = total - test_count
        train, test = None, None

        if self.split_strat in ["top", "mid", "bottom"]:
            df.sort_values(["row", "column", "scene"], inplace=True)
            if self.split_strat == "mid":
                train_portion_each_block = train_portion/2
                train_count_first_block = int(total * train_portion_each_block)

                train_first = df[0:train_count_first_block]
                test = df[train_count_first_block:train_count_first_block+test_count]
                train_second = df[train_count_first_block+test_count:]
                train = pd.concat([train_first, train_second])

            elif self.split_strat == "bottom":
                train = df[0: train_count]
                test = df[train_count:]

            elif self.split_strat == "top":
                test = df[0:test_count]
                train = df[test_count: ]

        elif self.split_strat in ["left", "right"]:
            df.sort_values(["column","row", "scene"], inplace=True)
            if self.split_strat == "right":
                train = df[0: train_count]
                test = df[train_count:]

            elif self.split_strat == "left":
                test = df[0:test_count]
                train = df[test_count:]
        elif self.split_strat == "block":
            df.sort_values(["row", "column", "scene"], inplace=True)
            max_row = df["row"].max()
            max_col = df["column"].max()
            block_len = 10
            block_x_count = int(max_row//block_len) + 1
            block_y_count = int(max_col//block_len) + 1
            blocks = []
            for x in range(block_x_count):
                for y in range(block_y_count):
                    blocks.append((x,y))

            indices = list(range(len(blocks)))
            count_sampled = int(len(indices)*test_portion)
            sampled_indices = random.sample(indices, count_sampled)
            test_indices = []
            for sample in sampled_indices:
                the_block = blocks[sample]
                block_row_start = the_block[0] * block_len
                block_row_end = block_row_start + block_len

                block_col_start = the_block[1] * block_len
                block_col_end = block_col_start + block_len
                current_df = df[ (df["row"] >=block_row_start) & (df["row"] < block_row_end)
                                 &(df["column"] >=block_col_start) & (df["column"] < block_col_end)
                                 ]
                for i in current_df.index:
                    test_indices.append(i)

            test = df.iloc[test_indices]
            train_indices = list(df.index)
            for i in test_indices:
                train_indices.remove(i)
            train = df.iloc[train_indices]

        return train, test



