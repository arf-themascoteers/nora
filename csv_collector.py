import os


class CSVCollector:
    @staticmethod
    def collect(source_dir):
        complete_csv_path = os.path.join(source_dir, "complete.csv")
        ag_csv_path = os.path.join(source_dir, "ag.csv")
        ml_csv_path = os.path.join(source_dir, "ml.csv")
        train_csv_path = os.path.join(source_dir, "train.csv")
        test_csv_path = os.path.join(source_dir, "test.csv")
        return complete_csv_path, ag_csv_path, ml_csv_path, train_csv_path, test_csv_path
        