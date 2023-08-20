import os
import pandas as pd
from environment import TEST
from hires1_csv_creator import Hires1CSVCreator
from csv_collector import CSVCollector
from hires1_scene_processor import Hires1SceneProcessor
from hires1_clips_to_df import Hires1ClipToDF


class Hires1Extractor:
    def __init__(self):
        self.TEST = TEST
        self.FILTERED = True
        self.source_csv = "vectis.csv"
        if self.TEST:
            self.source_csv = "vectis_min.csv"
        self.source_csv_path = os.path.join("data", self.source_csv)

        if self.FILTERED:
            short_csv = "shorter.csv"
            short_csv_path = os.path.join("data", short_csv)
            Hires1Extractor.shorten(self.source_csv_path, short_csv_path)
            self.source_csv_path = short_csv_path

        processed_dir = "hi1p"
        self.processed_dir_path = os.path.join("data", processed_dir)
        self.clip_path = os.path.join(self.processed_dir_path, "clipped")
        self.spatial_columns = ["scene", "row", "column"]

    @staticmethod
    def shorten(orig, short):
        df = pd.read_csv(orig)
        df = df[df["som"] > 1.72]
        df = df[df["som"] < 3.29]
        df.to_csv(short, index=False)

    def process(self):
        if os.path.exists(self.processed_dir_path):
            print(f"Dir exists for {self.processed_dir_path}. Skipping.")
            paths = CSVCollector.collect(self.processed_dir_path)
            return paths

        os.mkdir(self.processed_dir_path)
        os.mkdir(self.clip_path)
        scene_processor = Hires1SceneProcessor(self.clip_path, self.source_csv_path)
        dest = scene_processor.create_clips()
        cd = Hires1ClipToDF(dest, self.source_csv_path)
        df = cd.get_df()
        csv = Hires1CSVCreator(df, self.processed_dir_path, "low")
        paths = csv.create()
        return paths




