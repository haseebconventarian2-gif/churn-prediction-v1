import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class TestingPipeline:
    def run_testing(self, test_csv_path):
        # Load trained model
        model = joblib.load('artifacts/churn_model.pkl')

        # Load new data
        df = pd.read_csv(test_csv_path)

        # Drop target if present
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)

        # One-hot encode
        df = pd.get_dummies(df)

        # Align with training features
        train_df = pd.read_csv('artifacts/train.csv')
        train_df = pd.get_dummies(train_df.drop('Churn', axis=1))

        df = df.reindex(columns=train_df.columns, fill_value=0)

        # Scale
        scaler = StandardScaler()
        scaler.fit(train_df)
        df_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(df_scaled)

        return predictions