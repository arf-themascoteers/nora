import os
import pandas as pd
from haversine import haversine, Unit

os.chdir("../")

df = pd.read_csv("data/vectis.csv")
diss = []
for target in range(len(df)):
    lon_1000 = df.loc[target, "lon"]
    lat_1000 = df.loc[target, "lat"]
    point_1000 = (lat_1000, lon_1000)
    distance = 10000
    for i in range(len(df)):
        if i == target:
            continue
        lon = df.loc[i,"lon"]
        lat = df.loc[i,"lat"]
        point = (lat, lon)
        dis = haversine(point, point_1000)*1000
        if dis < distance:
            distance = dis

    print(distance)
    diss.append(distance)

diss = sorted(diss)
print(diss[0])
print(diss[-1])




