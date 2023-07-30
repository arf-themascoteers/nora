import os
import pandas as pd
import numpy as np
from clipper import clip
import shutil
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS
from rasterio.windows import Window
from sklearn.preprocessing import MinMaxScaler

SENTINEL_2_HOME = r"D:\Data\Tim\Created\Vectis\Sentinel-2"
EMPTY = -9999
TEST = True

def get_base(scene_path):
    safe = os.listdir(scene_path)[0]
    safe_path = os.path.join(scene_path, safe)
    granule_path = os.path.join(safe_path,"GRANULE")
    sub = os.listdir(granule_path)[0]
    sub_path = os.path.join(granule_path, sub)
    base = os.path.join(os.path.join(sub_path,"IMG_DATA"), "R60m")
    return base


def clip_bands(base, dest_clipped_scene_folder_path, source_csv_path):
    done = []
    for file_name in os.listdir(base):
        if not file_name.endswith(".jp2"):
            continue
        parts = file_name.split("_")
        band = parts[2]
        if band in done:
            continue
        done.append(band)
        source_band_path = os.path.join(base, file_name)
        dest_band_path = os.path.join(dest_clipped_scene_folder_path, f"{band}.tif")
        clip(source_band_path, dest_band_path, source_csv_path)
    return done


def iterate_bands(dest_clipped_scene_folder_path):
    for scene in os.listdir(dest_clipped_scene_folder_path):
        scene_path = os.path.join(dest_clipped_scene_folder_path, scene)
        for band in os.listdir(scene_path):
            if not band.endswith(".tif"):
                continue
            parts = band.split(".")
            band_part = parts[0]
            if band_part in ["AOT", "WVP", "SCL", "TCI"]:
                continue

            band_path = os.path.join(scene_path, band)
            with rasterio.open(band_path) as src:
                yield band_part, src


def create_table(dest_clipped_scene_folder_path, source_csv_path, scene_serial):
    epsg_4326 = CRS.from_epsg(4326)
    bands = []
    for band, src in iterate_bands(dest_clipped_scene_folder_path):
        bands.append(band)

    df = pd.read_csv(source_csv_path)
    df.drop(columns = ["when"], axis=1, inplace=True)
    all_columns = list(df.columns) + ["row","column","scene"] + bands
    table = np.zeros((len(df), len(all_columns)))
    data = df.to_numpy()
    table[:,0:data.shape[1]] = data[:,0:data.shape[1]]

    auxi_columns_count = len(df.columns)
    ROW_INDEX = auxi_columns_count
    COLUMN_INDEX = auxi_columns_count + 1
    SCENE_INDEX = auxi_columns_count + 2

    band_col = auxi_columns_count + 3

    for i in range(len(table)):
        table[i, SCENE_INDEX] = scene_serial
        table[i, ROW_INDEX] = EMPTY
        table[i, COLUMN_INDEX] = EMPTY

    for band, src in iterate_bands(dest_clipped_scene_folder_path):
        for i in range(len(table)):
            if i!=0 and i%1000 == 0:
                print(f"Done {i+1} ({table.shape[0]}) of {band_col+1} ({table.shape[1]})")
                if TEST:
                    break

            lon = table[i, 0]
            lat = table[i, 1]
            pixel_x, pixel_y = transform(epsg_4326, src.crs, [lon], [lat])
            pixel_x = round(pixel_x[0])
            pixel_y = round(pixel_y[0])
            row, column = src.index(pixel_x, pixel_y)
            row = min(row, src.shape[1]-1)
            column = min(column, src.shape[0]-1)

            if table[i,ROW_INDEX] == EMPTY:
                table[i,ROW_INDEX] = row

            if table[i,COLUMN_INDEX] == EMPTY:
                table[i,COLUMN_INDEX] = column

            window = Window(row, column, 1, 1)
            pixel_value = src.read(1, window=window)
            pixel_value = pixel_value[0,0]
            table[i,band_col] = pixel_value


        band_col = band_col + 1

    return table, all_columns


def make_ml_ready(dest_csv_path, ml_csv_path):
    df = pd.read_csv(dest_csv_path)
    df.drop(inplace=True, columns=["lat","lon"], axis=1)

    data = df.to_numpy()
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    df = pd.DataFrame(data= data, columns=df.columns)
    df.to_csv(ml_csv_path, index=False)


def process():
    TEST = False
    SKIP_CREATE_CLIP_DIRECTORY = False
    SKIP_CLIP = False
    SHORT = True

    source_csv = "vectis.csv"
    if TEST:
        source_csv = "vectis_min.csv"

    if SHORT:
        source_csv = "shorter.csv"

    source_csv_path = os.path.join("data", source_csv)
    dest_csv_path = os.path.join("data", "complete.csv")
    ml_csv_path = os.path.join("data", "ml.csv")
    dest_clipped_scene_folder_base_path = os.path.join("data", "clipped")

    if not SKIP_CREATE_CLIP_DIRECTORY:
        if os.path.exists(dest_clipped_scene_folder_base_path):
            shutil.rmtree(dest_clipped_scene_folder_base_path)

        os.mkdir(dest_clipped_scene_folder_base_path)
    if os.path.exists(dest_csv_path):
        os.remove(dest_csv_path)
    if os.path.exists(ml_csv_path):
        os.remove(ml_csv_path)
    df = None
    for scene_serial, scene in enumerate(os.listdir(SENTINEL_2_HOME)):
        scene_path = os.path.join(SENTINEL_2_HOME, scene)
        if not os.path.isdir(scene_path):
            continue
        if not SKIP_CLIP:
            base = get_base(scene_path)
            dest_clipped_scene_folder_path = os.path.join(dest_clipped_scene_folder_base_path, scene)
            os.mkdir(dest_clipped_scene_folder_path)
            clip_bands(base, dest_clipped_scene_folder_path, source_csv_path)

        table, columns = create_table(dest_clipped_scene_folder_base_path, source_csv_path, scene_serial)
        current_df = pd.DataFrame(data=table, columns=columns)
        if df is None:
            df = current_df
        else:
            df = pd.concat([df, current_df])
        print(f"Done {scene_serial}: {scene}")
        if TEST:
            break

    df.to_csv(dest_csv_path, index=False)
    make_ml_ready(dest_csv_path, ml_csv_path)


if __name__ == "__main__":
    process()



