import os
import sys
import pandas as pd
import numpy as np
import pickle

# Ensure project root is on sys.path so `from entity...` works when
# this file is executed directly (e.g. `python Components/data_transformation.py`).
# This inserts the parent directory of `Components/` (the project root).
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

class DataTransformationConfig:
    transformed_train_dir = os.path.join("artifacts", "transformed_train")
    transformed_test_dir = os.path.join("artifacts", "transformed_test")
    preprocessing_object_path = os.path.join("artifacts", "preprocessing.pkl")

class DataTransformation:
    def __init__(self, ingestion_artifact: DataIngestionArtifact, schema_path: str):
        self.ingestion_artifact = ingestion_artifact
        self.schema_path = schema_path
        self.config = DataTransformationConfig()

    def read_schema(self):
        import yaml
        with open(self.schema_path, "r") as file:
            return yaml.safe_load(file)

    def get_preprocessor(self, schema):
        num_features = schema.get("num_features", [])
        cat_features = schema.get("categorical_features", [])

        num_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])
        # Create an OneHotEncoder instance that requests dense output.
        # Different sklearn versions use either `sparse_output` or `sparse`.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                # Fallback: use default and accept sparse if necessary
                ohe = OneHotEncoder(handle_unknown="ignore")

        cat_pipeline = Pipeline([
            ("onehot", ohe)
        ])

        return ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        schema = self.read_schema()
        train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
        test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

        # Validate that schema feature names exist in the dataframes
        num_features = schema.get("num_features", [])
        cat_features = schema.get("categorical_features", [])
        expected_features = set(num_features + cat_features)
        missing_in_train = expected_features - set(train_df.columns)
        missing_in_test = expected_features - set(test_df.columns)
        if missing_in_train:
            raise ValueError(f"These features from schema are missing in train dataframe: {missing_in_train}")
        if missing_in_test:
            raise ValueError(f"These features from schema are missing in test dataframe: {missing_in_test}")

        # Coerce numeric features to numeric dtype, handle empty strings and spaces
        for col in num_features:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col].astype(str).str.strip().replace("", np.nan).replace(" ", np.nan), errors="coerce")
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col].astype(str).str.strip().replace("", np.nan).replace(" ", np.nan), errors="coerce")

        preprocessor = self.get_preprocessor(schema)

        X_train = train_df.drop(columns=["Churn"])
        y_train = train_df["Churn"]
        X_test = test_df.drop(columns=["Churn"])
        y_test = test_df["Churn"]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # If transformers returned sparse matrices convert to dense arrays
        try:
            # scipy sparse has .toarray()
            if hasattr(X_train_transformed, "toarray"):
                X_train_transformed = X_train_transformed.toarray()
            if hasattr(X_test_transformed, "toarray"):
                X_test_transformed = X_test_transformed.toarray()
        except Exception:
            pass

        # Attempt to obtain the output feature names from the fitted preprocessor.
        feature_names = None
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

        os.makedirs(self.config.transformed_train_dir, exist_ok=True)
        os.makedirs(self.config.transformed_test_dir, exist_ok=True)

        transformed_train_path = os.path.join(self.config.transformed_train_dir, "train_transformed.csv")
        transformed_test_path = os.path.join(self.config.transformed_test_dir, "test_transformed.csv")

        # Build DataFrame column names from preprocessor feature names when possible
        if feature_names is not None:
            try:
                cols_train = list(feature_names)
            except Exception:
                cols_train = [f"feature_{i}" for i in range(X_train_resampled.shape[1])]
        else:
            cols_train = [f"feature_{i}" for i in range(X_train_resampled.shape[1])]

        train_df_transformed = pd.DataFrame(X_train_resampled, columns=cols_train)
        train_df_transformed["Churn"] = y_train_resampled

        # For test set, use available feature names but align shapes if necessary
        if feature_names is not None and X_test_transformed.shape[1] == len(feature_names):
            cols_test = list(feature_names)
        else:
            cols_test = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

        test_df_transformed = pd.DataFrame(X_test_transformed, columns=cols_test)
        test_df_transformed["Churn"] = y_test.values

        train_df_transformed.to_csv(transformed_train_path, index=False)
        test_df_transformed.to_csv(transformed_test_path, index=False)

        with open(self.config.preprocessing_object_path, "wb") as file:
            pickle.dump(preprocessor, file)

        return DataTransformationArtifact(
            transformed_train_path=transformed_train_path,
            transformed_test_path=transformed_test_path,
            preprocessing_object_path=self.config.preprocessing_object_path)