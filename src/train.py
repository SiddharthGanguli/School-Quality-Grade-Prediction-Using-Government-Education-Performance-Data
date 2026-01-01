import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

class TrainModel:

    def __init__(self, split_dir, model_path, experiment_name="School_Grade_Prediction"):
        self.split_dir = split_dir
        self.model_path = model_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.experiment_name = experiment_name

    def load_splits(self):
        self.X_train = pd.read_csv(os.path.join(self.split_dir, "X_train.csv"))
        self.X_test = pd.read_csv(os.path.join(self.split_dir, "X_test.csv"))
        self.y_train = pd.read_csv(os.path.join(self.split_dir, "y_train.csv"))
        self.y_test = pd.read_csv(os.path.join(self.split_dir, "y_test.csv"))
    
    def train_model(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Model Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return acc, report

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved at {self.model_path}")

    def run_training(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            self.load_splits()
            self.train_model()
            acc, report = self.evaluate_model()
            self.save_model()

            # Log parameters
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            # Optionally log macro f1-score
            mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])
            mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

            # Log the model artifact
            mlflow.sklearn.log_model(self.model, "random_forest_model")
            mlflow.log_artifact(self.model_path)

            print("MLflow run successfully...")
