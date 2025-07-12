# Customer Churn Analysis Project

A beginner-friendly data analytics project that analyzes customer churn patterns using Python. This project demonstrates a complete data science pipeline from data loading to model deployment.

## 📋 Project Overview

This project analyzes customer churn data to understand:
- Which customers are most likely to leave the service
- Key factors that influence customer churn
- Patterns in customer behavior and demographics

## 🏗️ Project Structure

```
Sales_Analysis/
├── data/
│   ├── raw/
│   │   └── customer_churn.csv             # Original dataset
│   └── processed/
│       └── cleaned_churn_data.csv         # Cleaned version
├── notebooks/
│   └── churn_eda.ipynb                    # EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Functions to load raw/processed data
│   ├── preprocessing.py                   # Cleaning, missing values, encoding
│   ├── visualization.py                   # Plots like churn distribution, heatmaps
│   └── model.py                           # Logistic Regression or Decision Tree
├── reports/
│   └── figures/
│       ├── churn_by_contract_type.png
│       └── correlation_heatmap.png
├── tests/
│   └── test_preprocessing.py              # Unit tests for cleaning functions
├── config/
│   └── config.yaml                        # File paths, model parameters
├── requirements.txt                       # pandas, scikit-learn, matplotlib, etc.
├── README.md                              # Project summary, how to run, results
├── .gitignore                             # Ignore __pycache__/, .ipynb_checkpoints/
└── main.py                                # Entry point to run cleaning, modeling
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Sales_Analysis
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete analysis**
   ```bash
   python main.py
   ```

## 📊 Dataset Description

The project uses a sample customer churn dataset with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| customer_id | Unique customer identifier | Integer |
| age | Customer age | Integer |
| gender | Customer gender | Categorical |
| tenure | Number of months with the company | Integer |
| monthly_charges | Monthly service charges | Float |
| total_charges | Total charges to date | Float |
| contract_type | Type of contract | Categorical |
| payment_method | Payment method used | Categorical |
| churn | Whether customer left (target) | Binary |

## 🔧 Usage

### Running the Complete Pipeline

The main script (`main.py`) runs the entire analysis pipeline:

```bash
python main.py
```

This will:
1. Load the raw customer churn data
2. Clean and preprocess the data
3. Create visualizations and save them to `reports/figures/`
4. Train and evaluate machine learning models
5. Generate performance reports

### Individual Components

You can also run individual components:

**Data Loading:**
```python
from src.data_loader import load_raw_data
data = load_raw_data()
```

**Data Preprocessing:**
```python
from src.preprocessing import clean_data, encode_categorical_variables
cleaned_data = clean_data(data)
encoded_data, encoders = encode_categorical_variables(cleaned_data)
```

**Visualization:**
```python
from src.visualization import create_all_plots
create_all_plots(cleaned_data)
```

**Modeling:**
```python
from src.model import train_and_evaluate_models
predictor = train_and_evaluate_models(X_train, X_test, y_train, y_test)
```

### Jupyter Notebook

For interactive exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/churn_eda.ipynb
```

## 📈 Results

The analysis provides:

1. **Data Insights:**
   - Customer churn distribution
   - Key factors affecting churn
   - Correlation between features

2. **Visualizations:**
   - Churn distribution plots
   - Feature correlation heatmaps
   - Contract type analysis
   - Age and tenure distributions

3. **Model Performance:**
   - Logistic Regression and Decision Tree models
   - Accuracy and ROC AUC scores
   - Feature importance analysis

## 🧪 Testing

Run the unit tests to ensure data quality:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python tests/test_preprocessing.py
```

## 📁 Output Files

After running the analysis, you'll find:

- `data/processed/cleaned_churn_data.csv` - Cleaned and encoded dataset
- `reports/figures/` - All generated visualizations:
  - `churn_distribution.png`
  - `churn_by_contract_type.png`
  - `correlation_heatmap.png`
  - `age_distribution.png`
  - `monthly_charges_vs_tenure.png`
  - `confusion_matrix_*.png`
  - `roc_curves.png`

## 🔍 Key Findings

Based on the analysis:

1. **Contract Type Impact:** Month-to-month contracts have higher churn rates
2. **Tenure Effect:** Longer-tenured customers are less likely to churn
3. **Monthly Charges:** Higher charges correlate with increased churn risk
4. **Payment Method:** Electronic check users show different churn patterns

## 🛠️ Customization

### Configuration

Edit `config/config.yaml` to modify:
- File paths
- Model parameters
- Visualization settings

### Adding New Features

1. Add new features to the dataset
2. Update preprocessing functions in `src/preprocessing.py`
3. Modify feature selection in `prepare_features_target()`
4. Update tests accordingly

### Adding New Models

1. Create new model class in `src/model.py`
2. Add training method to `ChurnPredictor` class
3. Update the main pipeline in `main.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors:** Make sure you're in the project root directory
2. **Missing Dependencies:** Run `pip install -r requirements.txt`
3. **File Not Found:** Check that `data/raw/customer_churn.csv` exists
4. **Memory Issues:** Reduce dataset size for testing

### Getting Help

- Check the error messages for specific issues
- Review the test files for expected data formats
- Ensure all required packages are installed

## 📚 Learning Resources

This project demonstrates:
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Machine learning model training
- Model evaluation and comparison
- Data visualization
- Project organization and testing

Perfect for beginners learning data science with Python! 