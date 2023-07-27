import pandas as pd
import shapefile as shp
import pyproj


def get_bounding_box(source_csv_path):
  vpd = pd.read_csv(source_csv_path)
  # cnts = vpd.value_counts().reset_index(name='count')
  # cnts = cnts[cnts["count"] > 1]

  min_x = vpd["lon"].min()
  max_x = vpd["lon"].max()
  min_y = vpd["lat"].max()
  max_y = vpd["lat"].min()

  return min_x, min_y, max_x, max_y


if __name__ == "__main__":
  print(get_bounding_box())

