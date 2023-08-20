import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
import os
from rasterio.warp import transform
import re


class Hires1ClipToDF:
    def __init__(self, clipped, source_csv_path):
        self.clipped = clipped
        self.source_csv_path =  source_csv_path
        self.spatial_columns = ["row", "column"]
        self.bands = {"3":"B02","2":"B03", "1":"B04","4":"B08"}

    def get_df(self):
        epsg = self.get_epsg()
        bands = self.get_band_list()
        df = pd.read_csv(self.source_csv_path)
        df["when"] = Hires1ClipToDF.get_epoch(df["when"])
        all_columns = list(df.columns) + self.spatial_columns + bands
        table = np.zeros((len(df), len(all_columns)))
        data = df.to_numpy()
        table[:,0:data.shape[1]] = data[:,0:data.shape[1]]
        spatial_info_column_start = len(df.columns)
        band_index_start = spatial_info_column_start + len(self.spatial_columns)
        self.populate_scene_info(table, spatial_info_column_start)
        for column_offset, (band, src) in enumerate(self.iterate_bands()):
            column_index = band_index_start + column_offset
            for i in range(len(table)):
                if i!=0 and i%1000 == 0:
                    print(f"Done band processing {i+1} ({table.shape[0]}) of band {column_offset+1} ({len(bands)})")
                lon = table[i, 0]
                lat = table[i, 1]
                row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
                window = Window(column, row, 1, 1)
                pixel_value = src.read(1, window=window)
                pixel_value = pixel_value[0,0]
                table[i,column_index] = pixel_value


        df = pd.DataFrame(data=table, columns=all_columns)
        df.sort_values(self.spatial_columns, inplace=True)
        return df

    def get_epsg(self):
        return CRS.from_epsg(4326)

    @staticmethod
    def get_epoch(str_dates):
        return [int((datetime.strptime(str_date, '%d-%b-%y')).timestamp()) for str_date in str_dates]

    def get_band_list(self):
        return [band for band, src in self.iterate_bands()]

    def iterate_bands(self):
        with rasterio.open(self.clipped) as src:
            num_bands = src.count
            for band_index in range(1, num_bands + 1):
                yield self.bands[str(band_index)], src

    def populate_scene_info(self, table, start_index):
        epsg = self.get_epsg()
        ROW_INDEX = start_index
        COLUMN_INDEX = start_index + 1
        with rasterio.open(self.clipped) as src:
            for i in range(table.shape[0]):
                lon = table[i, 0]
                lat = table[i, 1]
                row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
                table[i, ROW_INDEX] = row
                table[i, COLUMN_INDEX] = column
                if i != 0 and i % 1000 == 0:
                    print(f"Done populating spatial {i + 1} of {table.shape[0]}")
        return table

    def get_row_col_by_lon_lat(self, epsg, src, lon, lat):
        pixel_x, pixel_y = transform(epsg, src.crs, [lon], [lat])
        row, column = src.index(pixel_x, pixel_y)
        row = row[0]
        column = column[0]
        row = min(row, src.height - 1)
        column = min(column, src.width - 1)
        return row, column
