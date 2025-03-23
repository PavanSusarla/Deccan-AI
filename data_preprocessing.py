import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_and_save_data():
    """Load California housing dataset and save it as CSV if not already saved."""
    data_path = "data/california_housing.csv"
    
    if not os.path.exists(data_path):  # Save only if the file doesn't exist
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df["MedHouseVal"] = california.target  # Add target column
        os.makedirs("data", exist_ok=True)  # Create 'data' folder if not exists
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    else:
        print(f"Dataset already exists at {data_path}")

def load_data():
    """Load the dataset and preprocess it."""
    data_path = "data/california_housing.csv"
    
    # Ensure the dataset is available
    load_and_save_data()
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Scale numerical features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Split dataset
    X = df_scaled.drop(columns=["MedHouseVal"])
    y = df_scaled["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)