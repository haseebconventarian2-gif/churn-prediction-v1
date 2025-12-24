import os
import sys
import pandas as pd

# Ensure project root is on sys.path so `from entity...` works when
# this file is executed directly (e.g. `python components/data_ingestion.py`).
# This inserts the parent directory of `components/` (the project root).
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from entity.artifact_entity import DataIngestionArtifact
from entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, file_path: str, config: DataIngestionConfig = DataIngestionConfig()):
        self.file_path = file_path
        self.config = config
        self.dataframe = None

    def read_data(self):
        self.dataframe = pd.read_csv(self.file_path)
        return self.dataframe

    def export_and_split(self):
        df = self.read_data()
        train_size = int(len(df) * (1 - self.config.train_test_split_ratio))
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df[:train_size]
        test_df = df[train_size:]

        os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)

        df.to_csv(self.config.feature_store_file_path, index=False)
        train_df.to_csv(self.config.train_file_path, index=False)
        test_df.to_csv(self.config.test_file_path, index=False)

        return DataIngestionArtifact(
            train_file_path=self.config.train_file_path,
            test_file_path=self.config.test_file_path,
            feature_store_path=self.config.feature_store_file_path,
        )
