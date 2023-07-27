import pandas as pd
import shapefile as shp
import pyproj

vpd = pd.read_csv("data/vectis.csv")
cnts = vpd.value_counts().reset_index(name='count')
cnts = cnts[cnts["count"] > 1]
#print(cnts)

w = shp.Writer('data/shapes/shapes', shapeType=shp.POLYGON)
w.field('field1', 'C')

min_x = vpd["lon"].min()
max_x = vpd["lon"].max()
min_y = vpd["lat"].max()
max_y = vpd["lat"].min()

w.poly([[
  [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
]])

w.record("poly1")

prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
with open("data/shapes/shapes.prj", 'w') as prj_file:
  prj_file.write(prj_content)

w.close()

