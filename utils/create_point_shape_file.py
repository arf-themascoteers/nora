import shapefile as shp
import pyproj
from get_bounding_box import get_bounding_box
import pandas as pd

w = shp.Writer('../data/all_shapes/filtered/filtered', shapeType=shp.POINT)
w.field("SOM", "F", 4,5)  # Field for the yield value (numeric, with a width of 10)


df = pd.read_csv(r"../data/shorter.csv")

for i in range(len(df)):
  w.point(df.loc[i,"lon"], df.loc[i,"lat"])
  w.record(df.loc[i,"som"])

prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
with open("../data/all_shapes/filtered/filtered.prj", 'w') as prj_file:
  prj_file.write(prj_content)

w.close()

print("done")

