import shapefile as shp
import pyproj
from get_bounding_box import get_bounding_box

w = shp.Writer('data/shapes/shapes', shapeType=shp.POLYGON)
w.field('field1', 'C')

min_x, min_y, max_x, max_y = get_bounding_box()

w.poly([[
  [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
]])

w.record("poly1")

prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
with open("data/shapes/shapes.prj", 'w') as prj_file:
  prj_file.write(prj_content)

w.close()

