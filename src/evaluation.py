"""
Model evaluation module for credit card approval prediction.
Includes ROC-AUC, precision-recall curves, and other metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


class ModelEvaluator:
    """
    Evaluate ML models for credit card approval prediction.
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name for the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, models_dict, X_test, y_test):
        """
        Evaluate all models and store results.
        
        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of all evaluation results
        """
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            self.evaluate_model(model, X_test, y_test, model_name)
        
        return self.results
    
    def print_metrics(self, model_name=None):
        """
        Print evaluation metrics.
        
        Args:
            model_name: Specific model name (prints all if None)
        """
        if model_name:
            models_to_print = {model_name: self.results[model_name]}
        else:
            models_to_print = self.results
        
        for name, metrics in models_to_print.items():
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print(f"{'='*50}")
            print(f"Accuracy:           {metrics['accuracy']:.4f}")
            print(f"Precision:          {metrics['precision']:.4f}")
            print(f"Recall:             {metrics['recall']:.4f}")
            print(f"F1 Score:           {metrics['f1_score']:.4f}")
            print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
            print(f"Avg Precision:      {metrics['avg_precision']:.4f}")
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
    
    def plot_roc_curves(self, y_test, figsize=(10, 8)):
        """
        Plot ROC curves for all evaluated models.
        
        Args:
            y_test: True test labels
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            auc = metrics['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Credit Card Approval Prediction', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, y_test, figsize=(10, 8)):
        """
        Plot Precision-Recall curves for all evaluated models.
        
        Args:
            y_test: True test labels
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, metrics in self.results.items():
            precision, recall, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
            avg_precision = metrics['avg_precision']
            plt.plot(recall, precision, 
                    label=f"{model_name} (AP = {avg_precision:.3f})")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Credit Card Approval Prediction', fontsize=14)
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_confusion_matrices(self, figsize=(15, 4)):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize: Figure size
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False)
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def compare_models(self):
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with model comparison
        """
        import pandas as pd
        
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'Avg Precision': metrics['avg_precision']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.round(4)
        
        return df_comparison


def evaluate_models(models_dict, X_test, y_test):
    """
    Convenience function to evaluate all models.
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        ModelEvaluator instance with results
    """
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(models_dict, X_test, y_test)
    
    return evaluator
