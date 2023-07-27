import pandas as pd

original = "../data/vectis.csv"
shorter = "../data/shorter.csv"
df = pd.read_csv(original)
CUT = False
if CUT:
    ndf = df[0:1]
    for i in range(1,len(df)):
        if i%5 == 0:
            ndf = pd.concat([ndf,df[i:i+1]])
    df = ndf

df = df[df["som"]>1.72]
df = df[df["som"]<3.29]

df.to_csv(shorter,index=False)
