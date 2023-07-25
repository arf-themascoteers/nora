import numpy as np
import rasterio as rio
from pyproj import Transformer

infile = "abc.tif"

with rio.open(infile) as src:
    transform = src.transform
    rows, cols = src.shape
    print(rows, cols)
    for c in range(cols):
        for r in range(rows):
            print(c,r)
            transformer = Transformer.from_crs(src.crs, "EPSG:4326")
            x, y = src.transform * (c,r)
            print(x,y)
            lon, lat = transformer.transform(x, y)
            print(lon, lat)
