"""
Data preprocessing functions for customer churn analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.data_loader import load_config


def check_missing_values(data):
    """
    Check for missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.Series: Missing values count for each column
    """
    missing_values = data.isnull().sum()
    print("🔍 Missing values check:")
    print(missing_values)
    return missing_values


def clean_data(data):
    """
    Clean the customer churn dataset.
    
    Args:
        data (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    print("🧹 Starting data cleaning...")
    
    # Create a copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Check for missing values
    missing_values = check_missing_values(cleaned_data)
    
    # Handle missing values (if any)
    if missing_values.sum() > 0:
        print("⚠️  Found missing values, handling them...")
        # For numeric columns, fill with median
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
    
    # Convert churn column to binary (Yes=1, No=0)
    cleaned_data['churn'] = cleaned_data['churn'].map({'Yes': 1, 'No': 0})
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['age', 'tenure', 'monthly_charges', 'total_charges']
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    print("✅ Data cleaning completed!")
    return cleaned_data


def encode_categorical_variables(data):
    """
    Encode categorical variables using Label Encoding.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with encoded categorical variables
    """
    print("🔤 Encoding categorical variables...")
    
    encoded_data = data.copy()
    label_encoders = {}
    
    # Identify categorical columns (excluding target variable)
    categorical_columns = ['gender', 'contract_type', 'payment_method']
    
    for column in categorical_columns:
        if column in encoded_data.columns:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(encoded_data[column])
            label_encoders[column] = le
            print(f"   Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print("✅ Categorical encoding completed!")
    return encoded_data, label_encoders


def prepare_features_target(data):
    """
    Prepare features and target variables for modeling.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print("🎯 Preparing features and target...")
    
    # Define features (exclude customer_id and target)
    feature_columns = ['age', 'gender', 'tenure', 'monthly_charges', 
                      'total_charges', 'contract_type', 'payment_method']
    
    # Select features and target
    X = data[feature_columns]
    y = data['churn']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Target distribution:\n{y.value_counts()}")
    
    return X, y


def split_data(X, y, config_path="config/config.yaml"):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        config_path (str): Path to configuration file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    config = load_config(config_path)
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def get_data_summary(data):
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'data_types': data.dtypes.to_dict(),
        'numeric_summary': data.describe().to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'categorical_counts': {}
    }
    
    # Get counts for categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        summary['categorical_counts'][col] = data[col].value_counts().to_dict()
    
    return summary


if __name__ == "__main__":
    # Test the preprocessing functions
    from src.data_loader import load_raw_data
    
    print("Testing preprocessing functions...")
    
    # Load raw data
    raw_data = load_raw_data()
    if raw_data is not None:
        # Clean data
        cleaned_data = clean_data(raw_data)
        
        # Encode categorical variables
        encoded_data, encoders = encode_categorical_variables(cleaned_data)
        
        # Prepare features and target
        X, y = prepare_features_target(encoded_data)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print("\n✅ All preprocessing functions tested successfully!") 