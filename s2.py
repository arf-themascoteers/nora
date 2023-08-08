import os
import pandas as pd
import numpy as np
import hashlib
from clipper import Clipper
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
from rasterio.windows import Window
from datetime import datetime
import re
from sklearn.preprocessing import MinMaxScaler


class S2Extractor:
    def __init__(self, ag="low", scenes=0):
        self.FILTERED = True
        self.SENTINEL_2_HOME = r"D:\Data\Tim\Created\Vectis\Sentinel-2"
        self.ag = ag

        if type(scenes) == list:
            self.scene_list = scenes
        else:
            self.scene_list = os.listdir(self.SENTINEL_2_HOME)
            self.scene_list = [scene for scene in self.scene_list if scene.startswith("S2")
                               and os.path.isdir(os.path.join(self.SENTINEL_2_HOME, scene))]
            if scenes == 0:
                scenes = len(self.scene_list)
            self.scene_list = self.scene_list[0:scenes]
        self.scene_list = sorted(self.scene_list)
        self.log_file_path = os.path.join("data","log.txt")
        self.log_file = open(self.log_file_path, "w")
        self.source_csv = "vectis_min.csv"
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

        self.scenes_str = S2Extractor.create_scenes_string(self.scene_list)
        self.dir_str_original = self.ag_str + "_"+self.scenes_str
        self.dir_hash =  hashlib.md5(self.dir_str_original.encode('UTF-8')).hexdigest()
        self.dir_hash_path = os.path.join(processed_dir, self.dir_hash)
        self.clip_path = os.path.join(self.dir_hash_path, "clipped")
        self.dest_csv_path = os.path.join(self.dir_hash_path, "complete.csv")
        self.ag_csv_path = os.path.join(self.dir_hash_path, "ag.csv")
        self.ml_csv_path = os.path.join(self.dir_hash_path, "ml.csv")

        self.spatial_columns = ["scene","row","column"]
        self.geo_columns = ["lon", "lat", "when"]

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
        if os.path.exists(self.datasets_list_file_path):
            df = pd.read_csv(self.datasets_list_file_path)
            df.loc[len(df)] = [dirname,ag,scenes]
            df.columns = ["dirname", "ag", "scenes"]
        else:
            df = pd.DataFrame(data=[[dirname,ag,scenes]], columns=["dirname", "ag", "scenes"])
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
        bands = sorted(bands, key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=self.is_reverve())

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
        SCENE_INDEX = start_index
        ROW_INDEX = start_index + 1
        COLUMN_INDEX = start_index + 2
        for i in range(len(table)):
            table[i, SCENE_INDEX] = scene_serial
        res = dict([(band, src.height * src.width) for band, src in self.iterate_bands(dest_clipped_scene_folder_path)])
        res = sorted(res.items(), key=lambda x: x[1], reverse=self.is_reverve())
        band = res[0][0]
        src = self.get_src_by_band(dest_clipped_scene_folder_path, band)

        for i in range(table.shape[0]):
            lon = table[i, 0]
            lat = table[i, 1]
            row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
            table[i, ROW_INDEX] = row
            table[i, COLUMN_INDEX] = column
            if i != 0 and i % 1000 == 0:
                print(f"Done populating spatial {i + 1} of {table.shape[0]} for scene {scene_serial}")

        return table

    def create_table(self, dest_clipped_scene_folder_path, scene_serial):
        epsg = self.get_epsg()
        bands = self.get_band_list(dest_clipped_scene_folder_path)
        df = pd.read_csv(self.source_csv_path)
        df["when"] = S2Extractor.get_epoch(df["when"])
        all_columns = list(df.columns) + self.spatial_columns + bands
        table = np.zeros((len(df), len(all_columns)))
        data = df.to_numpy()
        table[:,0:data.shape[1]] = data[:,0:data.shape[1]]
        spatial_info_column_start = len(df.columns)
        band_index_start = spatial_info_column_start + len(self.spatial_columns)
        self.populate_scene_info(table, dest_clipped_scene_folder_path, spatial_info_column_start, scene_serial)
        for column_offset, (band, src) in enumerate(self.iterate_bands(dest_clipped_scene_folder_path)):
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

    @staticmethod
    def create_scenes_string(scenes):
        return ":".join(scenes)

    def get_scene_source(self, scene):
        scene_path = os.path.join(self.SENTINEL_2_HOME, scene)
        return S2Extractor.get_base(scene_path)

    def get_scene_clip_folder_path(self, scene):
        return os.path.join(self.clip_path, scene)

    def create_clips(self):
        os.mkdir(self.dir_hash_path)
        os.mkdir(self.clip_path)
        for index, scene in enumerate(self.scene_list):
            dest_clipped_scene_folder_path = self.get_scene_clip_folder_path(scene)
            os.mkdir(dest_clipped_scene_folder_path)
            base = self.get_scene_source(scene)
            self.clip_bands(base, dest_clipped_scene_folder_path)
            print(f"Done clipping scene {index+1}: {scene}")
            self.log_file.write(f"{index+1},{scene}\n")

    def get_df_from_scenes(self):
        self.create_clips()
        df = None
        for index, scene in enumerate(self.scene_list):
            dest_clipped_scene_folder_path = self.get_scene_clip_folder_path(scene)
            table, columns = self.create_table(dest_clipped_scene_folder_path, index+1)
            current_df = pd.DataFrame(data=table, columns=columns)
            if df is None:
                df = current_df
            else:
                df = pd.concat([df, current_df])
            print(f"Done scene {index+1}: {scene}")
            self.log_file.write(f"{index+1},{scene}\n")
        df.sort_values(self.spatial_columns, inplace=True)
        return df

    def aggregate(self):
        df = pd.read_csv(self.dest_csv_path)
        df.drop(columns=self.geo_columns, axis=1, inplace=True)
        columns_to_agg = df.columns.drop(self.spatial_columns)
        if self.ag is not None:
            df = df.groupby(self.spatial_columns)[columns_to_agg].mean().reset_index()
        df.to_csv(self.ag_csv_path, index=False)

    def create_ml_ready_csv_from_df(self, df):
        df.to_csv(self.dest_csv_path, index=False)
        if self.ag:
            self.aggregate()
        self.make_ml_ready()
        self.log_file.close()
        self.write_dataset_list_file(self.dir_hash, self.ag_str, self.scenes_str)

    def process(self):
        if os.path.exists(self.dir_hash_path):
            print(f"Dir exists for {self.dir_str_original} - ({self.dir_hash_path}). Skipping.")
            return self.ml_csv_path, self.scene_list
        df = self.get_df_from_scenes()
        self.create_ml_ready_csv_from_df(df)
        return self.ml_csv_path, self.scene_list

    def make_ml_ready(self):
        df = pd.read_csv(self.ag_csv_path)
        df.drop(inplace=True, columns=self.spatial_columns, axis=1)
        for col in self.geo_columns:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        data = df.to_numpy()
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        df = pd.DataFrame(data=data, columns=df.columns)
        df.to_csv(self.ml_csv_path, index=False)

    def get_epsg(self):
        return CRS.from_epsg(4326)


if __name__ == "__main__":
    s2 = S2Extractor()
    s2.process()



