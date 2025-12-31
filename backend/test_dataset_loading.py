
from dataset_loader import DatasetLoader

def test_loading():
    print("Testing Dataset Loader...")
    loader = DatasetLoader()
    
    # Example: Loading the COVID-CT dataset
    # Note: ensure you have internet access and Kaggle credentials if needed
    try:
        dataset_handle = "luisblanche/covidct"
        print(f"Attempting to load: {dataset_handle}")
        
        # We leave file_path empty to let it decide or load the default file
        df = loader.load_dataset(dataset_handle)
        
        if df is not None:
            print("\nDataset Loaded Successfully!")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head())
        else:
            print("Failed to load dataframe.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_loading()
