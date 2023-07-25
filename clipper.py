import rasterio
from pyproj import Transformer

jp2_file_path = 'abc.tif'

top_left_lon, top_left_lat = 142.121, -36.733
bottom_right_lon, bottom_right_lat = 142.134, -36.747

with rasterio.open(jp2_file_path) as src:
    transformer = Transformer.from_crs("EPSG:4326", src.crs)
    top_left_x, top_left_y = transformer.transform(top_left_lon, top_left_lat)
    print(src.crs)
    exit(0)
    top_left_x, top_left_y = round(top_left_x), round(top_left_y)


    bottom_right_x, bottom_right_y = transformer.transform(bottom_right_lon, bottom_right_lat)
    bottom_right_x, bottom_right_y = round(bottom_right_x), round(bottom_right_y)
    print(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    rectangle = src.read(window=((top_left_y, bottom_right_y), (top_left_x, bottom_right_x)))
    rectangle_meta = src.meta.copy()
    rectangle_meta.update({
        'height': bottom_right_y - top_left_y,
        'width': bottom_right_x - top_left_x,
        'transform': rasterio.windows.transform(window=((top_left_y, bottom_right_y), (top_left_x, bottom_right_x)), transform=src.transform)
    })

output_file_path = 'out.jp2'
with rasterio.open(output_file_path, 'w', **rectangle_meta) as dst:
    dst.write(rectangle)

print("Rectangle extracted and saved as:", output_file_path)