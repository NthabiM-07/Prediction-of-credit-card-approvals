"""
Model training module for credit card approval prediction.
Includes baseline models and ensemble methods.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np


class ModelTrainer:
    """
    Train and manage ML models for credit card approval prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained model
        """
        params = {
            'random_state': self.random_state,
            'max_iter': 1000,
            **kwargs
        }
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_decision_tree(self, X_train, y_train, **kwargs):
        """
        Train Decision Tree model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for DecisionTreeClassifier
            
        Returns:
            Trained model
        """
        params = {
            'random_state': self.random_state,
            'max_depth': 5,
            **kwargs
        }
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['decision_tree'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train Random Forest ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained model
        """
        params = {
            'random_state': self.random_state,
            'n_estimators': 100,
            'max_depth': 10,
            **kwargs
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """
        Train Gradient Boosting ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for GradientBoostingClassifier
            
        Returns:
            Trained model
        """
        params = {
            'random_state': self.random_state,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            **kwargs
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all models: baseline and ensemble.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        print("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train)
        
        print("Training Decision Tree...")
        self.train_decision_tree(X_train, y_train)
        
        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        
        print("Training Gradient Boosting...")
        self.train_gradient_boosting(X_train, y_train)
        
        print(f"\nTrained {len(self.models)} models successfully!")
        return self.models
    
    def cross_validate_model(self, model, X, y, cv=5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Array of cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return scores
    
    def get_model(self, model_name):
        """
        Get a trained model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model or None if not found
        """
        return self.models.get(model_name)
    
    def get_all_models(self):
        """
        Get all trained models.
        
        Returns:
            Dictionary of all trained models
        """
        return self.models


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, stratify=y)
