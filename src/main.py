from preprocessing import PreprocessingData

def main():
    input_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Raw/datafile.csv"
    output_path = "/Users/siddharthaganguli/Desktop/Goverment/Data/Processed/processed_data.csv"

    preprocess = PreprocessingData(
        input_path=input_path,
        output_path=output_path
    )

    preprocess.run_preprocessing()
    print("Data preprocessing completed...")

if __name__ == "__main__":
    main()
