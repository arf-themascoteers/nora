from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def make_ml_ready(dest_csv_path, ml_csv_path):
    df = pd.read_csv(dest_csv_path)
    df.drop(inplace=True, columns=["row","column","scene"], axis=1)
    for col in ["lon","lat","when"]:
        if col in df.columns:
            df.drop(inplace=True, columns=[col], axis=1)
    data = df.to_numpy()
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    df = pd.DataFrame(data= data, columns=df.columns)
    df.to_csv(ml_csv_path, index=False)


if __name__ == "__main__":
    make_ml_ready("data/complete.csv","data/ml.csv")