from sklearn import model_selection
import pandas as pd
import random


class Splitter:
    def __init__(self, csv, mode="random", split="mid"):
        #mode = random, spatial
        #split = top, bottom, right, left, mid
        self.csv = csv
        self.mode = mode
        self.split = split

    @staticmethod
    def get_all_splits():
        return ["top", "bottom", "mid", "left", "right","block"]

    def split(self):
        df = pd.read_csv(self.csv)
        if self.mode == "random":
            return model_selection.train_test_split(df, test_size=0.2, random_state=2)

        total = len(df)
        test_portion = 0.2
        test_count = int(total*.2)
        train_portion = 1 - test_portion
        train_count = total - test_count
        train, test = None, None

        if self.split in ["top", "mid", "bottom"]:
            df.sort_values(["row", "column", "scene"], inplace=True)
            if self.split == "mid":
                train_portion_each_block = train_portion/2
                train_count_first_block = int(total * train_portion_each_block)

                train_first = df[0:train_count_first_block]
                test = df[train_count_first_block:train_count_first_block+test_count]
                train_second = df[train_count_first_block+test_count:]
                train = pd.concat([train_first, train_second])

            elif self.split == "bottom":
                train = df[0: train_count]
                test = df[train_count:]

            elif self.split == "top":
                test = df[0:test_count]
                train = df[test_count: ]

        elif self.split in ["left", "right"]:
            df.sort_values(["column","row", "scene"], inplace=True)
            if self.split == "right":
                train = df[0: train_count]
                test = df[train_count:]

            elif self.split == "left":
                test = df[0:test_count]
                train = df[test_count:]
        elif self.split == "block":
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



