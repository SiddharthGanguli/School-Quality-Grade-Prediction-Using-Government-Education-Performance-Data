import pandas as pd
import os
from sklearn.model_selection import train_test_split

class SplitData:

    def __init__(self, data_path, output_dir, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def split_features_target(self):
        self.X = self.df.drop(columns=['Grade_encoded'])
        self.y = self.df['Grade_encoded']
        return self.X, self.y

    def split_data(self):
        return train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def save_splits(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_dir, exist_ok=True)

        X_train.to_csv(os.path.join(self.output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.output_dir, "y_test.csv"), index=False)

    def run_split(self):
        self.load_data()
        self.split_features_target()

        X_train, X_test, y_train, y_test = self.split_data()
        self.save_splits(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test
