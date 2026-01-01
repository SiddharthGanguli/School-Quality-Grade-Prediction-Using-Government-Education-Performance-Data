from preprocessing import PreprocessingData
from split import SplitData
from train import TrainModel

def main():
    raw_data_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Raw/datafile.csv"
    processed_data_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Processed/processed_data.csv"
    split_output_dir = "/Users/siddharthaganguli/Desktop/Goverment/Data/Splits"
    model_path = "/Users/siddharthaganguli/Desktop/Goverment/model/grade_model.pkl"

    preprocess = PreprocessingData(raw_data_path, processed_data_path)
    preprocess.run_preprocessing()

    split = SplitData(
        data_path=processed_data_path,
        output_dir=split_output_dir
    )
    split.run_split()

    print("Data split into x_train and y_train.")


    trainer = TrainModel(split_output_dir, model_path)
    trainer.run_training()
    
if __name__ == "__main__":
    main()
