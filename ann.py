import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


class ANN(nn.Module):
    def __init__(self, device, train_ds, test_ds, alpha = 0.0):
        super().__init__()
        torch.manual_seed(1)
        self.TEST = False
        self.device = device
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.alpha = alpha
        self.num_epochs = 1000
        self.batch_size = 3000
        self.lr = 0.01

        x_size = train_ds.x.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(x_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')
        n_batches = int(len(self.train_ds)/self.batch_size) + 1

        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            batch_number = 0
            for (x, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                #print(f'Epoch:{epoch + 1} (of {self.num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    def test(self):
        batch_size = 100
        self.eval()
        self.to(self.device)

        dataloader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)

        y_all = np.zeros(0)
        y_hat_all = np.zeros(0)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_all = np.concatenate((y_all, y))
            y_hat_all = np.concatenate((y_hat_all, y_hat))

        r2 = r2_score(y_all, y_hat_all)
        return r2
