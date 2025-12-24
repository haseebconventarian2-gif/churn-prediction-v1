from sklearn.metrics import classification_report

class ModelEvaluation:
    def evaluate(self, model, X_test, y_test):
        report = classification_report(y_test, model.predict(X_test))
        return report