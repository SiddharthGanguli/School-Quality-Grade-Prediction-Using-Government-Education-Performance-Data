import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class TrainModel:

    def __init__(self, split_dir, model_path):
        self.split_dir = split_dir
        self.model_path = model_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

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
        report = classification_report(self.y_test, y_pred)

        print(f"Model Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved at {self.model_path}")

    def run_training(self):
        self.load_splits()
        self.train_model()
        self.evaluate_model()
        self.save_model()
