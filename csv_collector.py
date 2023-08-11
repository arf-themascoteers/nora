import os


class CSVCollector:
    @staticmethod
    def collect(source_dir):
        path = {
            "complete_csv_path": os.path.join(source_dir, "complete.csv"),
            "ag_csv_path": os.path.join(source_dir, "ag.csv"),
            "ml_csv_path": os.path.join(source_dir, "ml.csv"),
            "train_spatial_csv_path": os.path.join(source_dir, "train_spatial.csv"),
            "test_spatial_csv_path": os.path.join(source_dir, "test_spatial.csv"),
            "train_csv_path": os.path.join(source_dir, "train.csv"),
            "test_csv_path": os.path.join(source_dir, "test.csv")
        }
        return path
        