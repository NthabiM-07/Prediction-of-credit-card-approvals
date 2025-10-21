# Credit Card Approval Prediction

An end-to-end machine learning pipeline for predicting credit card approvals using the UCI Credit Card Approval dataset. This project demonstrates best practices in data science with a focus on reproducibility, modularity, and ethical AI considerations.

## 🎯 Project Overview

This project implements a complete machine learning workflow:
- **Data Processing**: Cleaning, imputation, encoding, and scaling
- **Baseline Models**: Logistic Regression and Decision Tree
- **Ensemble Methods**: Random Forest and Gradient Boosting
- **Evaluation**: ROC-AUC, Precision-Recall curves, and comprehensive metrics
- **Ethics**: Fairness and bias considerations for responsible AI

## 📁 Project Structure

```
Prediction-of-credit-card-approvals/
├── data/
│   └── cc_approvals.data          # Credit card approval dataset
├── notebooks/
│   └── credit_card_approval_prediction.ipynb  # Main analysis notebook
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Data cleaning and preprocessing
│   ├── model_training.py          # Model training modules
│   └── evaluation.py              # Model evaluation and metrics
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License information
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NthabiM-07/Prediction-of-credit-card-approvals.git
cd Prediction-of-credit-card-approvals
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data (if needed):
```bash
python src/data_loader.py
```

### Usage

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/credit_card_approval_prediction.ipynb
```

#### Option 2: Python Script
```python
from src import load_data, DataPreprocessor, ModelTrainer, ModelEvaluator, split_data

# Load data
df = load_data('data/cc_approvals.data')

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(df)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test, y_test)
evaluator.print_metrics()
```

## 📊 Dataset

The project uses a subset of the UCI Credit Card Approval dataset with the following characteristics:
- **Samples**: 690 credit card applications
- **Features**: 15 attributes (mix of categorical and numerical)
- **Target**: Binary classification (approved/denied)
- **Missing Values**: ~5% missingness handled through imputation

### Features
- Categorical features: Gender-related, employment status, credit history, etc.
- Numerical features: Age, income, credit score indicators, etc.
- All feature names are anonymized (A1-A15) to protect privacy

## 🔍 Methodology

### 1. Data Preprocessing
- **Missing Value Imputation**:
  - Numerical: Mean imputation
  - Categorical: Most frequent value imputation
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for feature normalization

### 2. Model Training
- **Baseline Models**:
  - Logistic Regression: Linear baseline with regularization
  - Decision Tree: Non-linear baseline with depth constraints
  
- **Ensemble Models**:
  - Random Forest: Bagging ensemble with 100 trees
  - Gradient Boosting: Boosting ensemble with adaptive learning

### 3. Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Probabilistic Metrics**: ROC-AUC, Average Precision
- **Visual Analysis**: ROC curves, Precision-Recall curves, Confusion matrices

## 📈 Results

Models are evaluated using multiple metrics to ensure robust performance assessment:

| Metric | Description | Importance |
|--------|-------------|------------|
| ROC-AUC | Area under ROC curve | Overall discrimination ability |
| Precision | Positive predictive value | Minimize false approvals |
| Recall | True positive rate | Capture all valid applicants |
| F1-Score | Harmonic mean of precision/recall | Balanced performance |

## ⚖️ Fairness and Bias Considerations

### Key Ethical Concerns

1. **Protected Attributes**: Age, gender, race may be correlated with features
2. **Disparate Impact**: Models may disproportionately affect certain groups
3. **Historical Bias**: Training data may reflect past discriminatory practices
4. **Transparency**: Applicants have right to understand denial reasons

### Recommended Practices

- **Regular Audits**: Monitor performance across demographic groups
- **Fairness Metrics**: Track demographic parity, equal opportunity
- **Explainability**: Use SHAP/LIME for decision explanation
- **Human Oversight**: Maintain human review for edge cases
- **Appeals Process**: Allow applicants to contest decisions

### Tools for Fairness Analysis
- [Fairlearn](https://fairlearn.org/) (Microsoft)
- [AI Fairness 360](https://aif360.mybluemix.net/) (IBM)
- [What-If Tool](https://pair-code.github.io/what-if-tool/) (Google)

## 🛠️ Development

### Running Tests
```bash
# Test data loading
python src/data_loader.py

# Test preprocessing
python -m pytest tests/  # (if tests are added)
```

### Code Style
The project follows PEP 8 style guidelines. Use `flake8` or `black` for formatting:
```bash
black src/
flake8 src/
```

## 📚 Dependencies

Core libraries:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models and preprocessing
- **matplotlib & seaborn**: Data visualization
- **jupyter**: Interactive notebooks

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Credit Card Approval dataset
- DataCamp for project inspiration and structure
- Open source community for excellent ML libraries

## 📧 Contact

For questions or feedback:
- GitHub: [@NthabiM-07](https://github.com/NthabiM-07)
- Project Issues: [Issue Tracker](https://github.com/NthabiM-07/Prediction-of-credit-card-approvals/issues)

## 🔮 Future Enhancements

- [ ] Hyperparameter optimization using GridSearchCV
- [ ] Additional ensemble methods (XGBoost, LightGBM, CatBoost)
- [ ] SMOTE for handling class imbalance
- [ ] Feature engineering and selection
- [ ] Cross-validation with multiple folds
- [ ] Model deployment with Flask/FastAPI
- [ ] Fairness-aware machine learning algorithms
- [ ] Interactive dashboard for predictions
- [ ] CI/CD pipeline for automated testing

---

**Note**: This is a demonstration project for educational purposes. Production deployment requires additional considerations including security, scalability, regulatory compliance, and comprehensive fairness audits.
