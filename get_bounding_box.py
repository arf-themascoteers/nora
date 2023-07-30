import pandas as pd
import shapefile as shp
import pyproj


def get_bounding_box(source_csv_path):
  PADDING = 0
  vpd = pd.read_csv(source_csv_path)
  min_x = vpd["lon"].min()
  max_x = vpd["lon"].max()
  min_y = vpd["lat"].max()
  max_y = vpd["lat"].min()
  min_x = min_x - PADDING
  max_x = max_x + PADDING
  min_y = min_y + PADDING
  max_y = max_y - PADDING
  return min_x, min_y, max_x, max_y


if __name__ == "__main__":
  print(get_bounding_box("data/vectis.csv"))

