"""
Credit Card Approval Prediction Pipeline

A reusable pipeline for predicting credit card approvals using machine learning.
"""

from .data_loader import load_data, create_sample_data
from .preprocessing import DataPreprocessor, preprocess_data
from .model_training import ModelTrainer, split_data
from .evaluation import ModelEvaluator, evaluate_models

__version__ = '1.0.0'

__all__ = [
    'load_data',
    'create_sample_data',
    'DataPreprocessor',
    'preprocess_data',
    'ModelTrainer',
    'split_data',
    'ModelEvaluator',
    'evaluate_models',
]
