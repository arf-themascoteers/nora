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


def clip_bands(base, dest_band_folder_path, source_csv_path):
    done = []
    for resolution in os.listdir(base):
        resolution_path = os.path.join(base, resolution)
        for file_name in os.listdir(resolution_path):
            if file_name.endswith(".jp2"):
                parts = file_name.split("_")
                band = parts[2]
                if band not in done:
                    done.append(band)
                    source_band_path = os.path.join(resolution_path, file_name)
                    dest_band_path = os.path.join(dest_band_folder_path, f"{band}.tif")
                    clip(source_band_path, dest_band_path, source_csv_path)
    return done


def iterate_bands(dest_band_folder_path):
    for band in os.listdir(dest_band_folder_path):
        if band.startswith("AOT"):
            continue
        if band.endswith(".tif"):
            band_path = os.path.join(dest_band_folder_path, band)
            with rasterio.open(band_path) as src:
                for subband in range(src.count):
                    yield band, subband, src.read(subband+1)


def create_table(dest_band_folder_path, source_csv_path):
    epsg_4326 = CRS.from_epsg(4326)
    bands = 0
    for band, subband, data in iterate_bands(dest_band_folder_path):
        bands = bands+1

    df = pd.read_csv(source_csv_path)
    df.drop(columns = ["when"], axis=1, inplace=True)
    cols = len(df.columns)
    new_cols = cols + bands
    table = np.zeros((len(df), new_cols))
    data = df.to_numpy()
    table[:,0:data.shape[1]] = data[:,0:data.shape[1]]

    for i in range(len(table)):
        current_col = cols
        lon = table[i,0]
        lat = table[i,1]
        for band in os.listdir(dest_band_folder_path):
            if band.startswith("AOT"):
                continue
            if band.endswith(".tif"):
                band_path = os.path.join(dest_band_folder_path, band)
                with rasterio.open(band_path) as src:
                    for subband in range(src.count):
                        pixel_x, pixel_y = transform(epsg_4326, src.crs, [lon], [lat])
                        pixel_x = round(pixel_x[0])
                        pixel_y = round(pixel_y[0])
                        row, column = src.index(pixel_x, pixel_y)
                        row = min(row, src.shape[1]-1)
                        column = min(column, src.shape[0]-1)
                        window = Window(row, column, 1, 1)
                        pixel_value = src.read(subband+1, window=window)
                        pixel_value = pixel_value[0,0]

                        table[i,current_col] = pixel_value
                        current_col = current_col + 1

    columns = list(df.columns)
    for band, subband, data in iterate_bands(dest_band_folder_path):
        columns.append(f"{band}_{subband}")
    return table, columns


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
    SKIP_CREATE = False
    source_csv = "vectis.csv"
    if TEST:
        source_csv = "vectis_min.csv"

    source_csv_path = os.path.join("data", source_csv)
    dest_csv_path = os.path.join("data", "complete.csv")
    ml_csv_path = os.path.join("data", "ml.csv")
    dest_band_folder_path = os.path.join("data", "bands")

    if not SKIP_CREATE:
        if os.path.exists(dest_band_folder_path):
            shutil.rmtree(dest_band_folder_path)

        os.mkdir(dest_band_folder_path)

    base = r"D:\Data\Sentinel-2\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA"

    source_df = pd.read_csv(source_csv_path)
    bands = clip_bands(base, dest_band_folder_path, source_csv_path)
    table, columns = create_table(dest_band_folder_path, source_csv_path)
    df = pd.DataFrame(data=table, columns=columns)
    df.to_csv(dest_csv_path, index=False)
    make_ml_ready(dest_csv_path, ml_csv_path)


if __name__ == "__main__":
    process()



