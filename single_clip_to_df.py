import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
import os
from rasterio.warp import transform
import re


class SingleClipsToDF:
    def __init__(self, clip_path, source_csv_path):
        self.clip_path = clip_path
        self.source_csv_path = source_csv_path
        self.spatial_columns = ["row", "column"]

    def get_df(self):
        table, columns = self.create_table()
        df = pd.DataFrame(data=table, columns=columns)
        df.sort_values(self.spatial_columns, inplace=True)
        return df

    def create_table(self):
        epsg = self.get_epsg()
        bands = self.get_band_list()
        df = pd.read_csv(self.source_csv_path)
        df["when"] = SingleClipsToDF.get_epoch(df["when"])
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

        return table, all_columns

    def get_epsg(self):
        return CRS.from_epsg(4326)

    @staticmethod
    def get_epoch(str_dates):
        return [int((datetime.strptime(str_date, '%d-%b-%y')).timestamp()) for str_date in str_dates]

    def get_band_list(self):
        return [band for band, src in self.iterate_bands()]

    def iterate_bands(self):
        bands = []
        for band in os.listdir(self.clip_path):
            if not band.endswith(".jp2"):
                continue
            parts = band.split(".")
            band_part = parts[0]
            if band_part[0] != 'B':
                continue
            bands.append(band_part)

        bands = sorted(bands, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for band in bands:
            file_name = f"{band}.jp2"
            band_path = os.path.join(self.clip_path, file_name)
            with rasterio.open(band_path) as src:
                yield band,src

    def populate_scene_info(self, table, start_index):
        epsg = self.get_epsg()
        ROW_INDEX = start_index
        COLUMN_INDEX = start_index + 1
        res = dict([(band, src.height * src.width) for band, src in self.iterate_bands()])
        res = sorted(res.items(), key=lambda x: x[1])
        band = res[0][0]
        src = self.get_src_by_band(self.clip_path, band)

        for i in range(table.shape[0]):
            lon = table[i, 0]
            lat = table[i, 1]
            row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
            table[i, ROW_INDEX] = row
            table[i, COLUMN_INDEX] = column
            if i != 0 and i % 1000 == 0:
                print(f"Done populating spatial {i + 1} of {table.shape[0]}")

        return table

    def get_src_by_band(self, dest_clipped_scene_folder_path, band):
        file_name = f"{band}.jp2"
        band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
        with rasterio.open(band_path) as src:
            return src

    def get_row_col_by_lon_lat(self, epsg, src, lon, lat):
        pixel_x, pixel_y = transform(epsg, src.crs, [lon], [lat])
        row, column = src.index(pixel_x, pixel_y)
        row = row[0]
        column = column[0]
        row = min(row, src.height - 1)
        column = min(column, src.width - 1)
        return row, column
