import pandas as pd

df = pd.read_csv("data/vectis.csv")
max_x = df[df["lon"] == df["lon"].max()]
max_y = df[df["lat"] == df["lat"].max()]
min_x = df[df["lon"] == df["lon"].min()]
min_y = df[df["lat"] == df["lat"].min()]
df = pd.concat([max_x, max_y, min_x, min_y])
df.to_csv("data/vectis_min.csv", index=False)
