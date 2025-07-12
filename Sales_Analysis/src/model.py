"""
Machine learning models for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_config


class ChurnPredictor:
    """
    A class to handle customer churn prediction using multiple models.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the ChurnPredictor.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        
    def train_logistic_regression(self, X_train, y_train):
        """
        Train a Logistic Regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            LogisticRegression: Trained model
        """
        print("🤖 Training Logistic Regression model...")
        
        lr_model = LogisticRegression(
            max_iter=self.config['model']['max_iter'],
            random_state=self.config['model']['random_state']
        )
        
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        print("✅ Logistic Regression model trained successfully!")
        return lr_model
    
    def train_decision_tree(self, X_train, y_train, max_depth=5):
        """
        Train a Decision Tree model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            max_depth (int): Maximum depth of the tree
            
        Returns:
            DecisionTreeClassifier: Trained model
        """
        print("🌳 Training Decision Tree model...")
        
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.config['model']['random_state']
        )
        
        dt_model.fit(X_train, y_train)
        self.models['decision_tree'] = dt_model
        
        print("✅ Decision Tree model trained successfully!")
        return dt_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a model and return performance metrics.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Performance metrics
        """
        print(f"📊 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store scores
        self.model_scores[model_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        
        return self.model_scores[model_name]
    
    def compare_models(self, X_test, y_test):
        """
        Compare all trained models and select the best one.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            str: Name of the best model
        """
        print("🏆 Comparing models...")
        
        best_score = 0
        best_model_name = None
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
            
            # Use ROC AUC as the primary metric
            score = self.model_scores[model_name]['roc_auc']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model = best_model_name
        print(f"🥇 Best model: {best_model_name} (ROC AUC: {best_score:.4f})")
        
        return best_model_name
    
    def plot_confusion_matrix(self, model, X_test, y_test, model_name, save_path=None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """
        Plot ROC curves for all models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curves saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, model_name='decision_tree'):
        """
        Get feature importance from the Decision Tree model.
        
        Args:
            model_name (str): Name of the model to get feature importance from
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': model.feature_names_in_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print(f"❌ Model {model_name} doesn't have feature importance!")
            return None
    
    def predict_churn(self, customer_data, model_name=None):
        """
        Predict churn for new customer data.
        
        Args:
            customer_data (pd.DataFrame): Customer features
            model_name (str): Name of the model to use (default: best model)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return None, None
        
        model = self.models[model_name]
        predictions = model.predict(customer_data)
        probabilities = model.predict_proba(customer_data)[:, 1]
        
        return predictions, probabilities


def train_and_evaluate_models(X_train, X_test, y_train, y_test, config_path="config/config.yaml"):
    """
    Train and evaluate multiple models for churn prediction.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        config_path (str): Path to configuration file
        
    Returns:
        ChurnPredictor: Trained predictor object
    """
    print("🚀 Starting model training and evaluation...")
    
    # Initialize predictor
    predictor = ChurnPredictor(config_path)
    
    # Train models
    predictor.train_logistic_regression(X_train, y_train)
    predictor.train_decision_tree(X_train, y_train)
    
    # Compare models
    best_model = predictor.compare_models(X_test, y_test)
    
    # Print detailed results
    print("\n📋 Detailed Results:")
    print("=" * 50)
    for model_name, scores in predictor.model_scores.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  ROC AUC: {scores['roc_auc']:.4f}")
        
        # Print classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, scores['predictions'], 
                                  target_names=['No Churn', 'Churn']))
    
    return predictor


if __name__ == "__main__":
    # Test the model functions
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data, encode_categorical_variables, prepare_features_target, split_data
    
    print("Testing model functions...")
    
    # Load and preprocess data
    raw_data = load_raw_data()
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        encoded_data, _ = encode_categorical_variables(cleaned_data)
        X, y = prepare_features_target(encoded_data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train and evaluate models
        predictor = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        print("\n✅ All model functions tested successfully!") 