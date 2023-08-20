import os
import pandas as pd
from environment import TEST
from single_csv_creator import SingleCSVCreator
from csv_collector import CSVCollector
from single_scene_processor import SingleSceneProcessor
from single_clip_to_df import SingleClipsToDF


class SingleS2Extractor:
    def __init__(self, scene):
        self.TEST = TEST
        self.FILTERED = True
        self.scene = scene
        self.source_csv = "vectis.csv"
        if self.TEST:
            self.source_csv = "vectis_min.csv"
        self.source_csv_path = os.path.join("data", self.source_csv)
        self.datasets_list_file = "datasets.csv"

        if self.FILTERED:
            short_csv = "shorter.csv"
            short_csv_path = os.path.join("data", short_csv)
            SingleS2Extractor.shorten(self.source_csv_path, short_csv_path)
            self.source_csv_path = short_csv_path

        processed_dir = "processed"
        self.processed_dir_path = os.path.join("data", processed_dir)

        if not os.path.exists(self.processed_dir_path):
            os.mkdir(self.processed_dir_path)

        self.scene_home = os.path.join(self.processed_dir_path, self.scene)
        self.clip_path = os.path.join(self.scene_home, "clipped")
        self.spatial_columns = ["scene", "row", "column"]

    @staticmethod
    def shorten(orig, short):
        df = pd.read_csv(orig)
        df = df[df["som"] > 1.72]
        df = df[df["som"] < 3.29]
        df.to_csv(short, index=False)

    def process(self):
        if os.path.exists(self.scene_home):
            print(f"Dir exists {self.scene_home}. Skipping.")
            paths = CSVCollector.collect(self.scene_home)
            return paths

        os.mkdir(self.scene_home)
        os.mkdir(self.clip_path)
        scene_processor = SingleSceneProcessor(self.scene, self.clip_path, self.source_csv_path)
        scene_processor.create_clips()
        cd = SingleClipsToDF(self.clip_path, self.source_csv_path)
        df = cd.get_df()
        csv = SingleCSVCreator(df, self.scene_home)
        paths = csv.create()
        return paths


if __name__ == "__main__":
    s2 = SingleS2Extractor()
    s2.process()



