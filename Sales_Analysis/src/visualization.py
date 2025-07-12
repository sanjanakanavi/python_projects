"""
Visualization functions for customer churn analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_config


def setup_plotting_style():
    """
    Set up the plotting style for consistent visualizations.
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_churn_distribution(data, save_path=None):
    """
    Plot the distribution of churn (target variable).
    
    Args:
        data (pd.DataFrame): Input data
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(8, 6))
    
    # Count churn values
    churn_counts = data['churn'].value_counts()
    
    # Create bar plot
    bars = plt.bar(['No Churn', 'Churn'], churn_counts.values, 
                   color=['lightblue', 'lightcoral'])
    
    # Add value labels on bars
    for bar, count in zip(bars, churn_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xlabel('Churn Status', fontsize=12)
    
    # Add percentage labels
    total = len(data)
    for i, (status, count) in enumerate(churn_counts.items()):
        percentage = (count / total) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', 
                ha='center', va='center', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Churn distribution plot saved to {save_path}")
    
    plt.show()


def plot_churn_by_contract_type(data, save_path=None):
    """
    Plot churn rate by contract type.
    
    Args:
        data (pd.DataFrame): Input data
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Calculate churn rate by contract type
    contract_churn = data.groupby('contract_type')['churn'].agg(['count', 'sum'])
    contract_churn['churn_rate'] = (contract_churn['sum'] / contract_churn['count']) * 100
    
    # Create bar plot
    bars = plt.bar(contract_churn.index, contract_churn['churn_rate'], 
                   color=['lightblue', 'lightgreen', 'lightcoral'])
    
    # Add value labels on bars
    for bar, rate in zip(bars, contract_churn['churn_rate']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold')
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.xlabel('Contract Type', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Contract type churn plot saved to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(data, save_path=None):
    """
    Plot correlation heatmap for numeric variables.
    
    Args:
        data (pd.DataFrame): Input data
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(10, 8))
    
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Correlation heatmap saved to {save_path}")
    
    plt.show()


def plot_age_distribution(data, save_path=None):
    """
    Plot age distribution by churn status.
    
    Args:
        data (pd.DataFrame): Input data
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall age distribution
    ax1.hist(data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Overall Age Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Age distribution by churn
    churned = data[data['churn'] == 1]['age']
    not_churned = data[data['churn'] == 0]['age']
    
    ax2.hist([not_churned, churned], bins=15, alpha=0.7, 
             label=['No Churn', 'Churn'], color=['lightblue', 'lightcoral'])
    ax2.set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Age distribution plot saved to {save_path}")
    
    plt.show()


def plot_monthly_charges_vs_tenure(data, save_path=None):
    """
    Plot monthly charges vs tenure with churn status.
    
    Args:
        data (pd.DataFrame): Input data
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    churned = data[data['churn'] == 1]
    not_churned = data[data['churn'] == 0]
    
    plt.scatter(not_churned['tenure'], not_churned['monthly_charges'], 
               alpha=0.6, label='No Churn', color='lightblue', s=50)
    plt.scatter(churned['tenure'], churned['monthly_charges'], 
               alpha=0.6, label='Churn', color='lightcoral', s=50)
    
    plt.title('Monthly Charges vs Tenure', fontsize=16, fontweight='bold')
    plt.xlabel('Tenure (months)', fontsize=12)
    plt.ylabel('Monthly Charges ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Monthly charges vs tenure plot saved to {save_path}")
    
    plt.show()


def create_all_plots(data, config_path="config/config.yaml"):
    """
    Create all visualization plots and save them.
    
    Args:
        data (pd.DataFrame): Input data
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    figures_dir = config['reports']['figures_dir']
    
    # Create figures directory if it doesn't exist
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    print("🎨 Creating all visualization plots...")
    
    # Create and save all plots
    plot_churn_distribution(data, f"{figures_dir}churn_distribution.png")
    plot_churn_by_contract_type(data, f"{figures_dir}churn_by_contract_type.png")
    plot_correlation_heatmap(data, f"{figures_dir}correlation_heatmap.png")
    plot_age_distribution(data, f"{figures_dir}age_distribution.png")
    plot_monthly_charges_vs_tenure(data, f"{figures_dir}monthly_charges_vs_tenure.png")
    
    print("✅ All plots created and saved successfully!")


if __name__ == "__main__":
    # Test the visualization functions
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data
    
    print("Testing visualization functions...")
    
    # Load and clean data
    raw_data = load_raw_data()
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        
        # Create all plots
        create_all_plots(cleaned_data)
        
        print("\n✅ All visualization functions tested successfully!") 