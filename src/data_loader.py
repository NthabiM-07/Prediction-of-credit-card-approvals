"""
Data loader for UCI Credit Card Approval dataset.
The dataset contains credit card applications with various features.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data(output_path='data/cc_approvals.data', n_samples=690):
    """
    Create sample credit card approval data similar to UCI dataset.
    
    The UCI Credit Approval dataset has 15 features (mix of categorical and numerical)
    and 1 target variable (+ or -).
    
    Feature description (anonymized in original):
    - A1: b, a (categorical)
    - A2: continuous
    - A3: continuous
    - A4: u, y, l, t (categorical)
    - A5: g, p, gg (categorical)
    - A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff (categorical)
    - A7: v, h, bb, j, n, z, dd, ff, o (categorical)
    - A8: continuous
    - A9: t, f (categorical)
    - A10: t, f (categorical)
    - A11: continuous
    - A12: t, f (categorical)
    - A13: g, p, s (categorical)
    - A14: continuous
    - A15: continuous
    - A16: +, - (target: approved or not)
    """
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        row = [
            np.random.choice(['b', 'a'], p=[0.7, 0.3]),  # A1
            np.random.normal(31.5, 12),  # A2
            np.random.exponential(4.5),  # A3
            np.random.choice(['u', 'y', 'l', 't'], p=[0.5, 0.25, 0.15, 0.1]),  # A4
            np.random.choice(['g', 'p', 'gg'], p=[0.6, 0.3, 0.1]),  # A5
            np.random.choice(['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff']),  # A6
            np.random.choice(['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o']),  # A7
            np.random.normal(2, 3),  # A8
            np.random.choice(['t', 'f'], p=[0.55, 0.45]),  # A9
            np.random.choice(['t', 'f'], p=[0.5, 0.5]),  # A10
            np.random.exponential(2.5),  # A11
            np.random.choice(['t', 'f'], p=[0.6, 0.4]),  # A12
            np.random.choice(['g', 'p', 's'], p=[0.7, 0.2, 0.1]),  # A13
            np.random.normal(180, 170),  # A14
            max(0, np.random.normal(1000, 2000)),  # A15
            np.random.choice(['+', '-'], p=[0.445, 0.555])  # A16 (target)
        ]
        data.append(row)
    
    # Add some missing values (represented as '?')
    for i in range(len(data)):
        for j in range(15):  # Don't add missing values to target
            if np.random.random() < 0.05:  # 5% missing
                data[i][j] = '?'
    
    # Create column names
    columns = [f'A{i}' for i in range(1, 16)] + ['A16']
    
    # Save to file
    with open(output_path, 'w') as f:
        for row in data:
            f.write(','.join([str(val) for val in row]) + '\n')
    
    print(f"Sample data created at {output_path}")
    return output_path


def load_data(filepath='data/cc_approvals.data'):
    """
    Load credit card approval data.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame with credit card application data
    """
    # Column names for the dataset
    columns = [f'A{i}' for i in range(1, 16)] + ['A16']
    
    # Load data
    df = pd.read_csv(filepath, names=columns, na_values='?', header=None)
    
    return df


if __name__ == '__main__':
    # Create sample data if file doesn't exist or is empty
    import os
    data_path = Path('data/cc_approvals.data')
    data_path.parent.mkdir(exist_ok=True, parents=True)
    
    if not data_path.exists() or os.path.getsize(data_path) == 0:
        create_sample_data(str(data_path))
    
    # Load and display basic info
    df = load_data(str(data_path))
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
