import os
import pandas as pd
import numpy as np
import hashlib
from clipper import Clipper
import shutil
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
from rasterio.windows import Window
from datetime import datetime
from aggregate import aggregate
import re
from sklearn.preprocessing import MinMaxScaler


class S2Extractor:
    def __init__(self, ag="low", scene=0):
        self.FILTERED = True
        self.SENTINEL_2_HOME = r"D:\Data\Tim\Created\Vectis\Sentinel-2"
        self.ag = ag

        if type(scene) == list:
            self.scene_list = scene
        else:
            self.scene_list = os.listdir(self.SENTINEL_2_HOME)
            self.scene_list = [scene for scene in self.scene_list if scene.startswith("S2")
                               and os.path.isdir(os.path.join(self.SENTINEL_2_HOME, scene))]
            if scene == 0:
                scene = len(self.scene_list)
            self.scene_list = self.scene_list[0:scene]
        self.scene_list = sorted(self.scene_list)
        self.log_file_path = os.path.join("data","log.txt")
        self.log_file = open(self.log_file_path, "w")
        self.source_csv = "vectis.csv"
        self.source_csv_path = os.path.join("data", self.source_csv)
        self.datasets_list_file = "datasets.csv"

        if self.FILTERED:
            short_csv = "shorter.csv"
            short_csv_path = os.path.join("data", short_csv)
            S2Extractor.shorten(self.source_csv_path, short_csv_path)
            self.source_csv_path = short_csv_path

        processed_dir = "processed"
        self.processed_dir_path = os.path.join("data", processed_dir)
        self.datasets_list_file_path = os.path.join(self.processed_dir_path,"datasets.csv")

        if not os.path.exists(self.processed_dir_path):
            os.mkdir(self.processed_dir_path)

        self.ag_str = "no_ag"
        if ag is not None:
            self.ag_str = ag

        self.scenes_str = ":".join(self.scene_list)
        self.dir_str_original = self.ag_str + "_"+self.scenes_str
        self.dir_hash =  hashlib.md5(self.dir_str_original.encode('UTF-8')).hexdigest()
        self.dir_hash_path = os.path.join(processed_dir, self.dir_hash)
        self.clip_path = os.path.join(self.dir_hash_path, "clipped")
        self.dest_csv_path = os.path.join(self.dir_hash_path, "complete.csv")
        self.ag_csv_path = os.path.join(self.dir_hash_path, "ag.csv")
        self.ml_csv_path = os.path.join(self.dir_hash_path, "ml.csv")

    @staticmethod
    def shorten(orig, short):
        df = pd.read_csv(orig)
        df = df[df["som"] > 1.72]
        df = df[df["som"] < 3.29]
        df.to_csv(short, index=False)

    def write_dataset_list_file(self, dirname, ag, scenes):
        row = self.read_dataset_list_file(dirname, ag, scenes)
        if row is not None:
            return
        if not os.path.exists(self.datasets_list_file_path):
            df = pd.read_csv(self.datasets_list_file_path)
            df.loc[len(df)] = [dirname,ag,scenes]
            df.columns = ["dirname", "ag", "scenes"]
        else:
            df = pd.DataFrame(data=[dirname,ag,scenes], columns=["dirname", "ag", "scenes"])
        df.to_csv(self.datasets_list_file_path, index=False)

    def read_dataset_list_file(self, dirname, ag, scenes):
        if not os.path.exists(self.datasets_list_file_path):
            return None

        df = pd.read_csv(self.datasets_list_file_path)
        df = df[((df['dirname'] == dirname) & (df['ag'] == ag) & (df['scenes'] == scenes))]
        if len(df) == 0:
            return None
        return df.iloc[0]

    @staticmethod
    def get_epoch(str_dates):
        return [int((datetime.strptime(str_date, '%d-%b-%y')).timestamp()) for str_date in str_dates]

    @staticmethod
    def get_base(scene_path):
        safe = os.listdir(scene_path)[0]
        safe_path = os.path.join(scene_path, safe)
        granule_path = os.path.join(safe_path,"GRANULE")
        sub = os.listdir(granule_path)[0]
        sub_path = os.path.join(granule_path, sub)
        img_path = os.path.join(os.path.join(sub_path,"IMG_DATA"))
        return img_path

    def clip_bands(self, base, dest_clipped_scene_folder_path):
        done = []
        folders = os.listdir(base)
        folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=self.is_reverve())
        for resolution in folders:
            resolution_path = os.path.join(base, resolution)
            for file_name in os.listdir(resolution_path):
                if not file_name.endswith(".jp2"):
                    continue
                parts = file_name.split("_")
                band = parts[2]
                if band in done:
                    continue
                done.append(band)
                source_band_path = os.path.join(resolution_path, file_name)
                dest_band_path = os.path.join(dest_clipped_scene_folder_path, f"{band}.jp2")
                clipper = Clipper(source_band_path, dest_band_path, self.source_csv_path)
                clipper.clip()
        return done

    def iterate_bands(self, dest_clipped_scene_folder_path):
        bands = []
        for band in os.listdir(dest_clipped_scene_folder_path):
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
            band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
            with rasterio.open(band_path) as src:
                yield band,src

    def get_src_by_band(self, dest_clipped_scene_folder_path, band):
        file_name = f"{band}.jp2"
        band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
        with rasterio.open(band_path) as src:
            return src

    def get_band_list(self, dest_clipped_scene_folder_path):
        return [band for band, src in self.iterate_bands(dest_clipped_scene_folder_path)]

    def is_reverve(self):
        if self.ag is not None and self.ag == "low":
            return True
        return False

    def get_row_col_by_lon_lat(self, epsg, src, lon, lat):
        pixel_x, pixel_y = transform(epsg, src.crs, [lon], [lat])
        row, column = src.index(pixel_x, pixel_y)
        row = row[0]
        column = column[0]
        row = min(row, src.height - 1)
        column = min(column, src.width - 1)
        return row, column

    def populate_scene_info(self, table, dest_clipped_scene_folder_path, start_index, scene_serial):
        epsg = self.get_epsg()
        ROW_INDEX = start_index
        COLUMN_INDEX = start_index + 1
        SCENE_INDEX = start_index + 2
        for i in range(len(table)):
            table[i, SCENE_INDEX] = scene_serial
        res = dict([(band, src.height * src.width) for band, src in self.iterate_bands(dest_clipped_scene_folder_path)])
        res = sorted(res.items(), key=lambda x: x[1], reverse=self.is_reverve())
        band = res[0][0]
        src = self.get_src_by_band(band)

        for i in range(table.shape[0]):
            lon = table[i, 0]
            lat = table[i, 1]
            row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
            table[i, ROW_INDEX] = row
            table[i, COLUMN_INDEX] = column

        return table

    def create_table(self, dest_clipped_scene_folder_path, scene_serial):
        epsg = self.get_epsg()
        spatial_info = ["row","column","scene"]
        bands = self.get_band_list(dest_clipped_scene_folder_path)
        df = pd.read_csv(self.source_csv_path)
        df["when"] = S2Extractor.get_epoch(df["when"])
        all_columns = list(df.columns) + spatial_info + bands
        table = np.zeros((len(df), len(all_columns)))
        data = df.to_numpy()
        table[:,0:data.shape[1]] = data[:,0:data.shape[1]]
        spatial_info_column_count = len(df.columns)
        band_index_start = spatial_info_column_count + len(spatial_info)

        for column_offset, (band, src) in enumerate(self.iterate_bands(dest_clipped_scene_folder_path)):
            column_index = band_index_start + column_offset
            for i in range(len(table)):
                if i!=0 and i%1000 == 0:
                    print(f"Done {i+1} ({table.shape[0]}) of {column_index+1} ({table.shape[1]})")
                lon = table[i, 0]
                lat = table[i, 1]
                row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
                window = Window(column, row, 1, 1)
                pixel_value = src.read(1, window=window)
                pixel_value = pixel_value[0,0]
                table[i,column_index] = pixel_value

        return table, all_columns

    def process(self):
        if os.path.exists(self.dir_hash_path):
            return self.ml_csv_path, self.scene_list
        os.mkdir(self.dir_hash_path)
        os.mkdir(self.clip_path)
        df = None
        scene_serial = 0
        for scene in self.scene_list:
            scene_path = os.path.join(self.SENTINEL_2_HOME, scene)
            base = S2Extractor.get_base(scene_path)
            dest_clipped_scene_folder_path = os.path.join(self.clip_path, scene)
            os.mkdir(dest_clipped_scene_folder_path)
            self.clip_bands(base, dest_clipped_scene_folder_path)
            scene_serial = scene_serial + 1
            table, columns = self.create_table(dest_clipped_scene_folder_path, scene_serial)
            current_df = pd.DataFrame(data=table, columns=columns)
            if df is None:
                df = current_df
            else:
                df = pd.concat([df, current_df])
            print(f"Done scene {scene_serial}: {scene}")
            self.log_file.write(f"{scene_serial},{scene}\n")

        df.to_csv(self.dest_csv_path, index=False)
        ml_source = self.dest_csv_path
        if self.ag:
            aggregate(self.dest_csv_path, self.ag_csv_path)
            ml_source = self.ag_csv_path
        self.make_ml_ready(ml_source, self.ml_csv_path)
        self.log_file.close()
        self.write_dataset_list_file(self.dir_hash, self.ag_str, self.scenes_str)
        return self.ml_csv_path, self.scene_list

    def make_ml_ready(self, source, ml_csv_path):
        df = pd.read_csv(source)
        df.drop(inplace=True, columns=["row", "column", "scene"], axis=1)
        for col in ["lon", "lat", "when"]:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        data = df.to_numpy()
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        df = pd.DataFrame(data=data, columns=df.columns)
        df.to_csv(ml_csv_path, index=False)

    def get_epsg(self):
        return CRS.from_epsg(4326)


if __name__ == "__main__":
    s2 = S2Extractor()
    s2.process()



