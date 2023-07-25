import rasterio
from pyproj import Transformer

jp2_file_path = 'abc.tif'

lat, lon = -36.73246686564262, 142.1206375420363
lat, lon = -36.74836891734051, 142.13576682724653

with rasterio.open(jp2_file_path) as src:
    t = Transformer.from_crs("EPSG:4326", src.crs)
    px, py = t.transform(lat, lon)
    print(px, py)
    inv_transform = ~src.transform
    print(inv_transform)
    x, y = inv_transform * (px, py)
    x = round(x)
    y = round(y)
    print(x, y)
    print(src.shape)
