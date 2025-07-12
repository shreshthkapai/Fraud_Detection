import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(filepath: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load and return the credit card dataset."""
    df = pd.read_csv(filepath)
    return df

def explore_data(df: pd.DataFrame) -> dict:
    """Basic exploration of the dataset."""
    stats = {
        'shape': df.shape,
        'fraud_count': df['Class'].value_counts().to_dict(),
        'fraud_percentage': (df['Class'].sum() / len(df)) * 100,
        'missing_values': df.isnull().sum().sum()
    }
    return stats

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning - this dataset is already clean."""
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Check for missing values (this dataset has none)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing data."""
    df = df.copy()

    # Time-based features
    df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
    df['Day'] = df['Time'] // (24 * 3600)
    
    # Amount-based features
    df['Amount_log'] = np.log1p(df['Amount']) 
    
    # Transaction velocity (transactions per hour for each day)
    df['Trans_per_hour'] = df.groupby('Day')['Time'].transform('count') / 24
    
    # --- ADD THIS LINE ---
    df = df.drop(columns=['Time']) # Drop the original, less useful Time column
    # ---------------------
    
    return df

def handle_outliers(df: pd.DataFrame, columns: list = ['Amount']) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def create_amount_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Create binned version of Amount feature."""
    df = df.copy()
    df['Amount_bin'] = pd.cut(df['Amount'], bins=n_bins, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['Amount_bin'] = df['Amount_bin'].astype(str)
    
    return df

def normalize_features(df: pd.DataFrame, features: list = ['Amount']) -> tuple:
    """Normalize specified features using MinMaxScaler."""
    df = df.copy()
    scaler = MinMaxScaler()
    
    df[features] = scaler.fit_transform(df[features])
    
    return df, scaler
