import pandas as pd


df = pd.DataFrame(data=[[1,2],[3,4],[5,6]])
df.insert(0,"c",pd.Series([10,20,20]))
df.loc[len(df)] = [10,20,30]
df.to_csv("test.csv", header=False)
print(df)