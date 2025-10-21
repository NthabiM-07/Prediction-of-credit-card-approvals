"""
Data preprocessing module for credit card approval prediction.
Includes cleaning, imputation, encoding, and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Preprocess credit card application data.
    """
    
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        
    def identify_column_types(self, df):
        """Identify numeric and categorical columns."""
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in df.columns:
            if col == 'A16':  # Skip target
                continue
            # Check if column can be converted to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                self.numeric_cols.append(col)
            except (ValueError, TypeError):
                self.categorical_cols.append(col)
        
        return self.numeric_cols, self.categorical_cols
    
    def clean_and_impute(self, df, fit=True):
        """
        Clean data and impute missing values.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the imputers (True for training, False for test)
            
        Returns:
            DataFrame with imputed values
        """
        df_clean = df.copy()
        
        # Identify column types if not done
        if not self.numeric_cols and not self.categorical_cols:
            self.identify_column_types(df_clean)
        
        # Convert numeric columns to float
        for col in self.numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Impute numeric columns
        if self.numeric_cols:
            if fit:
                df_clean[self.numeric_cols] = self.numeric_imputer.fit_transform(
                    df_clean[self.numeric_cols]
                )
            else:
                df_clean[self.numeric_cols] = self.numeric_imputer.transform(
                    df_clean[self.numeric_cols]
                )
        
        # Impute categorical columns
        if self.categorical_cols:
            if fit:
                df_clean[self.categorical_cols] = self.categorical_imputer.fit_transform(
                    df_clean[self.categorical_cols]
                )
            else:
                df_clean[self.categorical_cols] = self.categorical_imputer.transform(
                    df_clean[self.categorical_cols]
                )
        
        return df_clean
    
    def encode_categorical(self, df, fit=True):
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df_encoded[col] = df_encoded[col].astype(str)
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df, target_col='A16', fit=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            fit: Whether to fit transformers (True for training, False for test)
            
        Returns:
            X_scaled: Scaled feature matrix
            y: Target variable
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify column types on first run
        if fit and not self.numeric_cols and not self.categorical_cols:
            self.identify_column_types(X)
        
        # Clean and impute
        X_clean = self.clean_and_impute(X, fit=fit)
        
        # Encode categorical variables
        X_encoded = self.encode_categorical(X_clean, fit=fit)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit=fit)
        
        # Encode target variable
        if fit:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
        else:
            y_encoded = self.target_encoder.transform(y)
        
        return X_scaled, y_encoded


def preprocess_data(df, preprocessor=None, fit=True):
    """
    Convenience function for preprocessing.
    
    Args:
        df: Input DataFrame
        preprocessor: DataPreprocessor instance (creates new if None)
        fit: Whether to fit transformers
        
    Returns:
        X_scaled: Scaled features
        y: Encoded target
        preprocessor: The preprocessor instance used
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor()
    
    X_scaled, y = preprocessor.preprocess(df, fit=fit)
    
    return X_scaled, y, preprocessor
