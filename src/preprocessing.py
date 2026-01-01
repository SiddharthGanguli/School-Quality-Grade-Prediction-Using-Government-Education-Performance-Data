import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os 

class PreprocessingData:

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self.df

    def clean_column_names(self):
        self.df.columns = (
            self.df.columns
            .str.replace('\n', ' ', regex=False)
            .str.replace('.', '', regex=False)
            .str.strip()
        )
        return self.df

    def encode_target(self):
        self.df['Grade_encoded'] = self.label_encoder.fit_transform(self.df['Grade'])
        return self.df

    def drop_unnecessary_columns(self):
        self.df.drop(columns=['State', 'District', 'Grade'], inplace=True)
        return self.df

    def save_processed_data(self):
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.df.to_csv(self.output_path, index=False)

    def run_preprocessing(self):
        self.load_data()
        self.clean_column_names()
        self.encode_target()
        self.drop_unnecessary_columns()
        self.save_processed_data()

        return self.df
