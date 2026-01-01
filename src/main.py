from preprocessing import PreprocessingData
from split import SplitData
from train import TrainModel
from SMOTE_data import ApplySMOTE

def main():
    raw_data_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Raw/datafile.csv"
    processed_data_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Processed/processed_data.csv"
    split_output_dir = "/Users/siddharthaganguli/Desktop/Goverment/Data/Splits"
    model_path = "/Users/siddharthaganguli/Desktop/Goverment/model/grade_model.pkl"
    smote_split="/Users/siddharthaganguli/Desktop/Goverment/Data/Smote"

    preprocess = PreprocessingData(raw_data_path, processed_data_path)
    preprocess.run_preprocessing()

    split = SplitData(
        data_path=processed_data_path,
        output_dir=split_output_dir
    )
    split.run_split()

    print("Data split into x_train and y_train.")


    smote_data=ApplySMOTE(
        split_dir=split_output_dir,
        output_dir=split_output_dir

    )
    smote_data.run_smote()

    # trainer = TrainModel(split_output_dir, model_path,experiment_name="School_Grade_Prediction")
    # trainer.run_training()

    # trainer = TrainModel(split_output_dir, model_path,experiment_name="SmoteSchool_Grade_Prediction")
    # trainer.run_training()
    

if __name__ == "__main__":
    main()
