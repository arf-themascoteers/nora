import os
import pandas as pd
import numpy as np
from clipper import clip
import shutil

TEST = True
source_csv = "vectis.csv"
if TEST:
    source_csv = "vectis_min.csv"

source_csv_path = os.path.join("data",source_csv)
dest_csv_path = os.path.join("data","final.csv")
dest_band_folder_path = os.path.join("data", "bands")

if os.path.exists(dest_csv_path):
    shutil.rmtree(dest_band_folder_path)

os.mkdir(dest_band_folder_path)

base = r"D:\Data\Sentinel-2\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA"

source_df = pd.read_csv(source_csv_path)
num_rows = source_df.shape[0]
data = np.zeros((num_rows,22))

done = []


def clip_band(source_band_path, dest_band_path):
    clip(source_band_path, dest_band_path)


def process():
    for resolution in os.listdir(base):
        resolution_path = os.path.join(base, resolution)
        for file_name in os.listdir(resolution_path):
            if file_name.endswith(".jp2"):
                parts = file_name.split("_")
                band = parts[2]
                if band not in done:
                    done.append(band)
                    source_band_path = os.path.join(resolution_path, file_name)
                    dest_band_path = os.path.join(dest_band_folder_path, band)
                    clip_band(source_band_path, dest_band_path, source_csv_path)


if __name__ == "__main__":
    process()



