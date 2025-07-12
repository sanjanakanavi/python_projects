"""
Data loading functions for customer churn analysis.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_raw_data(config_path="config/config.yaml"):
    """
    Load raw customer churn data.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        pd.DataFrame: Raw customer churn data
    """
    config = load_config(config_path)
    raw_data_path = config['data']['raw']
    
    try:
        data = pd.read_csv(raw_data_path)
        print(f"✅ Successfully loaded raw data from {raw_data_path}")
        print(f"📊 Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {raw_data_path}")
        return None


def load_processed_data(config_path="config/config.yaml"):
    """
    Load processed (cleaned) customer churn data.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        pd.DataFrame: Processed customer churn data
    """
    config = load_config(config_path)
    processed_data_path = config['data']['processed']
    
    try:
        data = pd.read_csv(processed_data_path)
        print(f"✅ Successfully loaded processed data from {processed_data_path}")
        print(f"📊 Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {processed_data_path}")
        return None


def save_processed_data(data, config_path="config/config.yaml"):
    """
    Save processed data to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    processed_data_path = config['data']['processed']
    
    # Create directory if it doesn't exist
    Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(processed_data_path, index=False)
    print(f"✅ Successfully saved processed data to {processed_data_path}")


if __name__ == "__main__":
    # Test the data loading functions
    print("Testing data loading functions...")
    
    # Load raw data
    raw_data = load_raw_data()
    if raw_data is not None:
        print("\nRaw data preview:")
        print(raw_data.head())
        print(f"\nData types:\n{raw_data.dtypes}") 