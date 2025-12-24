import os
import sys

# Ensure project root is on sys.path so `from entity...` works when this
# script is executed directly.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from entity.artifact_entity import DataIngestionArtifact
from components.data_transformation import DataTransformation

def main():
    ingestion = DataIngestionArtifact(
        train_file_path='artifacts/train.csv',
        test_file_path='artifacts/test.csv',
        feature_store_path='artifacts/feature_store.csv'
    )
    dt = DataTransformation(ingestion, schema_path='config/schema.yaml')
    artifact = dt.initiate_data_transformation()
    print('DataTransformation completed:')
    print(artifact)

if __name__ == '__main__':
    main()
