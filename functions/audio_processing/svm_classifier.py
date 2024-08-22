from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os


class SVMClassifier:
    def __init__(self, model_directory):
        self.models = {}
        self.scalers = {}

        for filename in os.listdir(model_directory):
            if filename.endswith(".pkl"):
                model_name = filename.split(".")[0]
                model_path = os.path.join(model_directory, filename)
                model_data = joblib.load(model_path)
                self.models[model_name] = model_data["model"]
                self.scalers[model_name] = model_data["scaler"]

    def classify(self, features):
        results = {}

        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]
            results[model_name] = prediction

        return results


# Model training and saving script
def save_model(model, scaler, model_path):
    joblib.dump({"model": model, "scaler": scaler}, model_path)
