"""
Main entry point for Customer Churn Analysis project.
This script runs the complete pipeline: data loading, preprocessing, visualization, and modeling.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import load_raw_data, save_processed_data
from src.preprocessing import clean_data, encode_categorical_variables, prepare_features_target, split_data
from src.visualization import create_all_plots
from src.model import train_and_evaluate_models


def main():
    """
    Main function to run the complete customer churn analysis pipeline.
    """
    print("🚀 Starting Customer Churn Analysis Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Load raw data
        print("\n📂 Step 1: Loading raw data...")
        raw_data = load_raw_data()
        
        if raw_data is None:
            print("❌ Failed to load raw data. Exiting...")
            return
        
        print(f"✅ Raw data loaded successfully! Shape: {raw_data.shape}")
        
        # Step 2: Data cleaning and preprocessing
        print("\n🧹 Step 2: Data cleaning and preprocessing...")
        cleaned_data = clean_data(raw_data)
        
        # Encode categorical variables
        encoded_data, label_encoders = encode_categorical_variables(cleaned_data)
        
        # Save processed data
        save_processed_data(encoded_data)
        
        print("✅ Data preprocessing completed!")
        
        # Step 3: Create visualizations
        print("\n🎨 Step 3: Creating visualizations...")
        create_all_plots(cleaned_data)  # Use cleaned_data for better plots
        print("✅ Visualizations created and saved!")
        
        # Step 4: Prepare data for modeling
        print("\n🎯 Step 4: Preparing data for modeling...")
        X, y = prepare_features_target(encoded_data)

        # added by me
        valid = y.notna()
        X = X[valid]
        y = y[valid]

        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print("✅ Data prepared for modeling!")
        
        # Step 5: Train and evaluate models
        print("\n🤖 Step 5: Training and evaluating models...")
        predictor = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        print("✅ Model training and evaluation completed!")
        
        # Step 6: Generate additional model visualizations
        print("\n📊 Step 6: Generating model visualizations...")
        
        # Plot confusion matrices
        for model_name, model in predictor.models.items():
            predictor.plot_confusion_matrix(
                model, X_test, y_test, model_name,
                f"reports/figures/confusion_matrix_{model_name}.png"
            )
        
        # Plot ROC curves
        predictor.plot_roc_curve(X_test, y_test, "reports/figures/roc_curves.png")
        
        # Get feature importance (for Decision Tree)
        feature_importance = predictor.get_feature_importance()
        if feature_importance is not None:
            print("\n🔍 Feature Importance (Decision Tree):")
            print(feature_importance)
        
        print("✅ Model visualizations completed!")
        
        # Step 7: Summary
        print("\n📋 Pipeline Summary:")
        print("=" * 50)
        print(f"✅ Raw data loaded: {raw_data.shape}")
        print(f"✅ Data cleaned and encoded: {encoded_data.shape}")
        print(f"✅ Training samples: {X_train.shape[0]}")
        print(f"✅ Testing samples: {X_test.shape[0]}")
        print(f"✅ Models trained: {len(predictor.models)}")
        print(f"✅ Best model: {predictor.best_model}")
        
        if predictor.best_model:
            best_scores = predictor.model_scores[predictor.best_model]
            print(f"✅ Best model accuracy: {best_scores['accuracy']:.4f}")
            print(f"✅ Best model ROC AUC: {best_scores['roc_auc']:.4f}")
        
        print("\n🎉 Customer Churn Analysis Pipeline completed successfully!")
        print("\n📁 Generated files:")
        print("   - data/processed/cleaned_churn_data.csv")
        print("   - reports/figures/*.png")
        
    except Exception as e:
        print(f"❌ Error occurred during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 