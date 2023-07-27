import shapefile as shp
import pyproj
from get_bounding_box import get_bounding_box
import pandas as pd

w = shp.Writer('../data/long_shape/long_shape', shapeType=shp.POINT)
w.field("SOM", "F", 4,5)  # Field for the yield value (numeric, with a width of 10)


df = pd.read_csv(r"C:\Users\kisho\OneDrive\Desktop\complete2.csv")

for i in range(len(df)):
  w.point(df.loc[i,"lon"], df.loc[i,"lat"])
  w.record(df.loc[i,"som"])

prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
with open("../data/long_shape/long_shape.prj", 'w') as prj_file:
  prj_file.write(prj_content)

w.close()

print("done")

