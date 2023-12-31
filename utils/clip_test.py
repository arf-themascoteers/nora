import rasterio
from get_bounding_box import get_bounding_box
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import math


def clip(source, dest, source_csv_path):
    PADDING = 10
    min_x, min_y, max_x, max_y = get_bounding_box(source_csv_path)
    with rasterio.open(source) as src:
        epsg_4326 = CRS.from_epsg(4326)
        min_x, min_y, max_x, max_y = transform_bounds(epsg_4326, src.crs, min_x, max_y, max_x, min_y)

        (column, row) = (~src.transform) * (min_x, max_y)
        row = math.floor(row)
        column = math.floor(column)

        row = max(0,row-PADDING)
        column = max(0,column-PADDING)

        (column_max, row_max) = (~src.transform) * (max_x, min_y)
        row_max = math.ceil(row_max)
        column_max = math.ceil(column_max)

        row_max = min(src.height-1, row_max+PADDING)
        column_max = min(src.width-1, column_max+PADDING)

        height = row_max - row
        width = column_max - column



        #window = src.window(min_x, min_y, max_x, max_y)
        window = Window(column, row, width, height)

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


if __name__ == "__main__":
    source = r"D:\Data\Tim\Created\Vectis\Sentinel-2\S2B_MSIL2A_20220503T002659_N0400_R016_T54HXE_20220503T023159\S2B_MSIL2A_20220503T002659_N0400_R016_T54HXE_20220503T023159.SAFE\GRANULE\L2A_T54HXE_A026926_20220503T003625\IMG_DATA\R10m\T54HXE_20220503T002659_B02_10m.jp2"
    dest = r"D:\out\abc.tif"
    source_csv_path = "../data/shorter.csv"
    clip(source, dest, source_csv_path)
