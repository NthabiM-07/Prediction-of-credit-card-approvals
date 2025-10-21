"""
Demo script for Credit Card Approval Prediction Pipeline

This script demonstrates the complete workflow using the reusable modules.
"""

from src import (
    load_data,
    DataPreprocessor,
    ModelTrainer,
    split_data,
    ModelEvaluator
)
import warnings
warnings.filterwarnings('ignore')


def main():
    """Run the complete credit card approval prediction pipeline."""
    
    print("=" * 60)
    print("Credit Card Approval Prediction Demo")
    print("=" * 60)
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    df = load_data('data/cc_approvals.data')
    print(f"  ✓ Loaded {df.shape[0]} samples with {df.shape[1]} features")
    print(f"  ✓ Target distribution: {df['A16'].value_counts().to_dict()}")
    print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
    
    # 2. Preprocess data
    print("\n[Step 2] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    print(f"  ✓ Processed features shape: {X.shape}")
    print(f"  ✓ Target encoded classes: {set(y)}")
    
    # 3. Split data
    print("\n[Step 3] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    print(f"  ✓ Training samples: {X_train.shape[0]}")
    print(f"  ✓ Test samples: {X_test.shape[0]}")
    
    # 4. Train models
    print("\n[Step 4] Training models...")
    trainer = ModelTrainer(random_state=42)
    
    print("  • Logistic Regression...", end=" ")
    logreg = trainer.train_logistic_regression(X_train, y_train)
    print(f"✓ (Train acc: {logreg.score(X_train, y_train):.3f})")
    
    print("  • Decision Tree...", end=" ")
    tree = trainer.train_decision_tree(X_train, y_train)
    print(f"✓ (Train acc: {tree.score(X_train, y_train):.3f})")
    
    print("  • Random Forest...", end=" ")
    rf = trainer.train_random_forest(X_train, y_train)
    print(f"✓ (Train acc: {rf.score(X_train, y_train):.3f})")
    
    print("  • Gradient Boosting...", end=" ")
    gb = trainer.train_gradient_boosting(X_train, y_train)
    print(f"✓ (Train acc: {gb.score(X_train, y_train):.3f})")
    
    # 5. Evaluate models
    print("\n[Step 5] Evaluating models on test set...")
    evaluator = ModelEvaluator()
    models = trainer.get_all_models()
    results = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # 6. Display results
    print("\n[Step 6] Model Comparison Results:")
    print("-" * 60)
    comparison_df = evaluator.compare_models()
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['ROC-AUC'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_roc_auc = comparison_df.loc[best_model_idx, 'ROC-AUC']
    
    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC Score: {best_roc_auc:.4f}")
    print("=" * 60)
    
    print("\n✅ Pipeline execution completed successfully!")
    print("\nNext steps:")
    print("  1. Run the Jupyter notebook for detailed analysis and visualizations")
    print("  2. Review fairness and bias considerations in the documentation")
    print("  3. Consider hyperparameter tuning for improved performance")
    
    return evaluator, trainer, preprocessor


if __name__ == '__main__':
    evaluator, trainer, preprocessor = main()
