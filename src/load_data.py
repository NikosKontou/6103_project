import pandas as pd
import os

def load_raw_data(data_path="data/raw/AmesHousing.csv"):
    """
    Load the Ames Housing dataset from a CSV file.
    
    Parameters:
        data_path (str): Relative or absolute path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")
    return pd.read_csv(data_path)