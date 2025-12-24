import os
import sys

# When this file is executed directly `python pipeline/training_pipeline.py`,
# Python sets sys.path[0] to the `pipeline/` directory which prevents
# top-level package imports like `components` from being resolved. Add the
# project root (parent of `pipeline/`) to `sys.path` so `components` is importable.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation
from components.model_pusher import ModelPusher

class TrainingPipeline:
    def run_pipeline(self, file_path):
        ingestion = DataIngestion(file_path)
        ingestion_artifact = ingestion.initiate_data_ingestion()

        validation = DataValidation()
        validation.validate(ingestion_artifact.train_path)

        transformation = DataTransformation()
        transformed = transformation.transform(
            ingestion_artifact.train_path,
            ingestion_artifact.test_path
        )

        trainer = ModelTrainer()
        trainer_artifact = trainer.train(
            transformed.X_train,
            transformed.y_train,
            transformed.X_test,
            transformed.y_test
        )

        evaluator = ModelEvaluation()
        report = evaluator.evaluate(
            trainer_artifact.model,
            transformed.X_test,
            transformed.y_test
        )

        pusher = ModelPusher()
        pusher.save_model(trainer_artifact.model)

        return report