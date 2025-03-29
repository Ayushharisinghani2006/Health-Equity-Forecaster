import pandas as pd
import numpy as np

def clean_data(file_path):
    """Clean the dataset and perform feature engineering."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # Check for required columns
    required_columns = ["Region", "Population", "Hospitals", "Disease_Rate"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Remove outliers (e.g., Hospitals > Population)
    df = df[df["Hospitals"] <= df["Population"]]
    
    # Feature engineering
    df["Hospitals_per_Capita"] = df["Hospitals"] / df["Population"]
    df["Disease_Burden"] = df["Disease_Rate"] * df["Population"]
    
    return df

if __name__ == "__main__":
    df = clean_data("data/health_data.csv")
    print("Processed Data:\n", df.head())