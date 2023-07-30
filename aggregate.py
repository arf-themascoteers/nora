import pandas as pd


def aggregate(complete_csv, ag_csv):
    df = pd.read_csv(complete_csv)
    df.drop(columns=["lon","lat","when"], axis=1, inplace=True)
    columns_to_agg = df.columns.drop(["row", "column","scene"])
    df = df.groupby(["row","column","scene"])[columns_to_agg].mean().reset_index()
    df.to_csv(ag_csv, index=False)


if __name__ == "__main__":
    aggregate("data/complete.csv","data/ag.csv")