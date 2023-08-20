import os
from base_path import HI_RES1_PATH
from clipper import Clipper


class Hires1SceneProcessor:
    def __init__(self, clip_path, source_csv_path):
        self.clip_path = clip_path
        self.source_csv_path = source_csv_path

    def create_clips(self):
        dest_band_path = os.path.join(self.clip_path, f"clipped.tif")
        clipper = Clipper(HI_RES1_PATH, dest_band_path, self.source_csv_path, padding=100)
        clipper.clip()
        return dest_band_path
