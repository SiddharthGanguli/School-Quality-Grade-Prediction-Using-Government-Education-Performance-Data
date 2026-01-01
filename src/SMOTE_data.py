import pandas as pd
import os
from imblearn.over_sampling import SMOTE

class ApplySMOTE:

    def __init__(self, split_dir, output_dir, random_state=42):
        self.split_dir = split_dir
        self.output_dir = output_dir
        self.random_state = random_state

        self.X_train = None
        self.y_train = None

    def load_splits(self):
        self.X_train = pd.read_csv(os.path.join(self.split_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.split_dir, "y_train.csv")).squeeze()

    def apply_smote(self):
        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"After SMOTE, training set shape: {self.X_train.shape}, {self.y_train.shape}")

    def save_smote_data(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.X_train.to_csv(os.path.join(self.output_dir, "X_train_smote.csv"), index=False)
        self.y_train.to_csv(os.path.join(self.output_dir, "y_train_smote.csv"), index=False)
        print(f"Oversampled data saved in {self.output_dir}")

    def run_smote(self):
        self.load_splits()
        self.apply_smote()
        self.save_smote_data()
