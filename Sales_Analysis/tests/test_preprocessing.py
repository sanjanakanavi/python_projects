"""
Unit tests for preprocessing functions.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import clean_data, encode_categorical_variables, prepare_features_target


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'tenure': [12, 24, 6, 36, 18],
            'monthly_charges': [50.0, 75.5, 30.0, 90.0, 60.0],
            'total_charges': [600.0, 1812.0, 180.0, 3240.0, 1080.0],
            'contract_type': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
            'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
            'churn': ['No', 'No', 'Yes', 'No', 'Yes']
        })
    
    def test_clean_data(self):
        """Test data cleaning function."""
        cleaned = clean_data(self.test_data)
        
        # Check that churn is converted to binary
        self.assertTrue(all(cleaned['churn'].isin([0, 1])))
        
        # Check that numeric columns are properly typed
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['tenure']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['monthly_charges']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['total_charges']))
        
        # Check that no data is lost
        self.assertEqual(len(cleaned), len(self.test_data))
    
    def test_encode_categorical_variables(self):
        """Test categorical variable encoding."""
        cleaned = clean_data(self.test_data)
        encoded, encoders = encode_categorical_variables(cleaned)
        
        # Check that categorical columns are encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['gender']))
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['contract_type']))
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['payment_method']))
        
        # Check that encoders are created
        self.assertIn('gender', encoders)
        self.assertIn('contract_type', encoders)
        self.assertIn('payment_method', encoders)
        
        # Check that no data is lost
        self.assertEqual(len(encoded), len(cleaned))
    
    def test_prepare_features_target(self):
        """Test feature and target preparation."""
        cleaned = clean_data(self.test_data)
        encoded, _ = encode_categorical_variables(cleaned)
        X, y = prepare_features_target(encoded)
        
        # Check feature shape
        expected_features = ['age', 'gender', 'tenure', 'monthly_charges', 
                           'total_charges', 'contract_type', 'payment_method']
        self.assertEqual(list(X.columns), expected_features)
        
        # Check target shape
        self.assertEqual(len(y), len(encoded))
        
        # Check that target is binary
        self.assertTrue(all(y.isin([0, 1])))
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        data_with_missing.loc[1, 'gender'] = np.nan
        
        cleaned = clean_data(data_with_missing)
        
        # Check that missing values are handled
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
    
    def test_data_types_consistency(self):
        """Test that data types are consistent after preprocessing."""
        cleaned = clean_data(self.test_data)
        encoded, _ = encode_categorical_variables(cleaned)
        
        # Check numeric columns
        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(encoded[col]))
        
        # Check categorical columns are encoded as numeric
        categorical_cols = ['gender', 'contract_type', 'payment_method']
        for col in categorical_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(encoded[col]))


if __name__ == '__main__':
    unittest.main() 