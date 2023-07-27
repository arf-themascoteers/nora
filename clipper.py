import rasterio
from get_bounding_box import get_bounding_box
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.crs import CRS


def clip(source, dest, source_csv_path):
    min_x, min_y, max_x, max_y = get_bounding_box(source_csv_path)
    with rasterio.open(source) as src:
        epsg_4326 = CRS.from_epsg(4326)
        min_x, min_y, max_x, max_y = transform_bounds(epsg_4326, src.crs, min_x, max_y, max_x, min_y)
        window = src.window(min_x, min_y, max_x, max_y)
        data = src.read(window=window)
        profile = src.profile
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform),
            'nodata': None
        })
        with rasterio.open(dest, 'w', **profile) as dst:
            dst.write(data)
