import os

ARTIFACT_DIR = os.path.join(os.getcwd(), "artifacts")

class DataIngestionConfig:
    train_test_split_ratio: float = 0.2
    feature_store_file_path: str = os.path.join(ARTIFACT_DIR, "feature_store.csv")
    train_file_path: str = os.path.join(ARTIFACT_DIR, "train.csv")
    test_file_path: str = os.path.join(ARTIFACT_DIR, "test.csv")
