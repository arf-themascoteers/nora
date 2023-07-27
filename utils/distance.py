import os
import pandas as pd
from haversine import haversine, Unit

os.chdir("../")

df = pd.read_csv("data/vectis.csv")

for i in range(len(df)-1):
    lon = df.loc[i,"lon"]
    lat = df.loc[i,"lat"]

    lon2 = df.loc[i+1,"lon"]
    lat2 = df.loc[i+1,"lat"]

    dis = haversine((lat, lon), (lat2, lon2))*1000
    print(dis)



