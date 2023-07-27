import numpy as np
import rasterio as rio
from pyproj import Transformer

infile = "out.tif"


with rio.open(infile) as src:
    transform = src.transform
    print(transform)
    rows, cols = src.shape
    top_left_x, top_left_y = transform * (0, 0)
    print(top_left_x, top_left_y)
    bottom_right_x, bottom_right_y = transform * (cols, rows)

transformer = Transformer.from_crs(src.crs, "EPSG:4326")
top_left_lon, top_left_lat = transformer.transform(top_left_x, top_left_y)
bottom_right_lon, bottom_right_lat = transformer.transform(bottom_right_x, bottom_right_y)
print(top_left_x, top_left_y)
print("Top-Left Corner (Longitude, Latitude):", top_left_lon, top_left_lat)
print("Bottom-Right Corner (Longitude, Latitude):", bottom_right_lon, bottom_right_lat)