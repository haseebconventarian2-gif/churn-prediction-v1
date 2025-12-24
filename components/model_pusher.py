import joblib

class ModelPusher:
    def save_model(self, model):
        joblib.dump(model, 'artifacts/churn_model.pkl')