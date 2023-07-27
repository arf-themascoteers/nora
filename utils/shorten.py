import pandas as pd

original = "../data/vectis.csv"
shorter = "../data/shorter.csv"
df = pd.read_csv(original)
ndf = df[0:1]
for i in range(len(df)):
    if i!=0 and i%5 == 0:
        ndf = pd.concat([ndf,df[i:i+1]])

ndf = ndf[ndf["som"]>1.72]
ndf = ndf[ndf["som"]<3.29]

ndf.to_csv(shorter,index=False)
