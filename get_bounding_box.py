import pandas as pd
import shapefile as shp
import pyproj


def get_bounding_box():
  vpd = pd.read_csv("data/vectis.csv")
  # cnts = vpd.value_counts().reset_index(name='count')
  # cnts = cnts[cnts["count"] > 1]

  min_x = vpd["lon"].min()
  max_x = vpd["lon"].max()
  min_y = vpd["lat"].max()
  max_y = vpd["lat"].min()

  return min_x, min_y, max_x, max_y

