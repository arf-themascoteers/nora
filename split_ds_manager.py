import pandas as pd
import torch
from sklearn import model_selection


class SplitDSManager:
    def __init__(self, train=None, test=None, x=None, y=None):
        torch.manual_seed(0)
        train_df = pd.read_csv(train)
        test_df = pd.read_csv(test)
        x = x
        y = y
        if y is None:
            y = "som"
        if x is None:
            x = list(train_df.columns)
            x.remove(y)
        x_columns = x
        y_columns = [y]
        columns = x_columns + y_columns
        train_df = train_df[columns]
        test_df = test_df[columns]
        train_df = train_df.sample(frac=1)
        test_df = test_df.sample(frac=1)
        train_np = train_df.to_numpy()
        test_np = test_df.to_numpy()

        train_np, validation_np = model_selection.train_test_split(train_np, test_size=0.1, random_state=2)

        self.train_x = train_np[:,:-1]
        self.train_y = train_np[:,-1]

        self.test_x = test_np[:,:-1]
        self.test_y = test_np[:, -1]

        self.validation_x = validation_np[:,:-1]
        self.validation_y = validation_np[:, -1]

    def get_datasets(self):
        return self.train_x, self.train_y, self.test_x, self.test_y, self.validation_x, self.validation_y

