import os
import sys
import pickle
import yaml
import pandas as pd

# Ensure project root is on sys.path so `from entity...` works when
# this file is executed directly (e.g. `python Components/model_trainer.py`).
# This inserts the parent directory of `Components/` (the project root).
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from entity.artifact_entity import ModelTrainerArtifact

def read_transformed_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def train_and_evaluate_models(train_df, test_df):
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier()
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    model_path = "./artifacts/best_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    report = {
        "best_model": best_name,
        "accuracy": best_score,
        "precision": precision_score(y_test, best_model.predict(X_test), average="weighted"),
        "recall": recall_score(y_test, best_model.predict(X_test), average="weighted"),
        "f1_score": f1_score(y_test, best_model.predict(X_test), average="weighted")
    }

    yaml_path = "./artifacts/model_report.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(report, f)

    return best_model, report


class ModelTrainer:
    """Class wrapper expected by the training pipeline.

    The `train` method trains candidate models, selects the best one by
    accuracy on the test set, saves the trained model and report to
    `./artifacts/` and returns a `ModelTrainerArtifact` object. The
    returned artifact will also have a `.model` attribute referencing the
    trained model instance to maintain compatibility with existing
    pipeline code that expects `trainer_artifact.model`.
    """

    def __init__(self, artifacts_dir: str = "./artifacts"):
        self.artifacts_dir = artifacts_dir

    def train(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier()
        }

        best_model = None
        best_score = -1.0
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name

        os.makedirs(self.artifacts_dir, exist_ok=True)
        model_path = os.path.join(self.artifacts_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        report = {
            "best_model": best_name,
            "accuracy": best_score,
            "precision": precision_score(y_test, best_model.predict(X_test), average="weighted"),
            "recall": recall_score(y_test, best_model.predict(X_test), average="weighted"),
            "f1_score": f1_score(y_test, best_model.predict(X_test), average="weighted")
        }

        yaml_path = os.path.join(self.artifacts_dir, "model_report.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(report, f)

        artifact = ModelTrainerArtifact(
            trained_model_file_path=model_path,
            report_artifact=yaml_path
        )

        # Attach the live model instance for compatibility with existing code
        setattr(artifact, "model", best_model)

        return artifact
