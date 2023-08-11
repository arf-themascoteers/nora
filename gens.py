import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


df = pd.read_csv("complete.csv")
cols = df.columns
data = df.to_numpy()
for i in range(len(df.columns)):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(data[:,i].reshape(-1,1))
    data[:, i] = np.squeeze(x_scaled)
df = pd.DataFrame(data=data, columns=cols)
df.to_csv("proc.csv",index=False)

