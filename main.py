from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.data_validation import DataValidator, load_csv_data, load_schema
from components.model_trainer import read_transformed_data, train_and_evaluate_models
from entity.config_entity import DataIngestionConfig

def run_training_pipeline():
    # CSV in main folder
    csv_path = "./WA_Fn-UseC_-Telco-Customer-Churn.csv"
    schema_path = "./config/schema.yaml"

    # 1️⃣ Data Ingestion
    data_ingestion = DataIngestion(csv_path, DataIngestionConfig())
    diArtifacts = data_ingestion.export_and_split()

    # 2️⃣ Data Validation
    test_csv_path = diArtifacts.test_file_path
    data = load_csv_data(test_csv_path)
    schema = load_schema(schema_path)
    validator = DataValidator(data, schema)
    validator.validate()
    validator.summary()

    # 3️⃣ Data Transformation
    data_transformation = DataTransformation(diArtifacts, schema_path)
    transformation_artifacts = data_transformation.initiate_data_transformation()

    # 4️⃣ Model Training
    train_df, test_df = read_transformed_data(transformation_artifacts.transformed_train_path,
                                              transformation_artifacts.transformed_test_path)
    best_model, report = train_and_evaluate_models(train_df, test_df)
    print("Best model:", report["best_model"])
    print("Model accuracy:", report["accuracy"])

if __name__ == "__main__":
    run_training_pipeline()
