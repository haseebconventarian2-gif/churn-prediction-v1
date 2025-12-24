from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    feature_store_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    preprocessing_object_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    report_artifact: str
