import pandas as pd
import shapefile as shp
import pyproj


df = pd.read_csv(r"D:\work\nora\x.csv")
df.sort_values(["Longitude","Latitude"],inplace=True)

w = shp.Writer(r'D:\work\nora\shp\my', shapeType=shp.POINT)
w.field("SOM", "F", 4,5)


for i in range(len(df)):
  w.point(df.loc[i,"Longitude"], df.loc[i,"Latitude"])
  w.record(df.loc[i,"som"])

prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
with open(r'D:\work\nora\shp\my.prj', 'w') as prj_file:
  prj_file.write(prj_content)

w.close()

print("done")

