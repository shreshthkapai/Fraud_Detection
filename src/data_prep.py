import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Load credit card fraud dataset and perform basic validation.
    
    Args:
        file_path: Path to the creditcard.csv file
        
    Returns:
        pd.DataFrame: Loaded and validated dataset
    """
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
            
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Perform basic exploration of the fraud dataset.
    
    Args:
        df: Credit card fraud dataset
        
    Returns:
        dict: Dataset statistics and class distribution
    """
    stats = {
        'shape': df.shape,
        'total_transactions': len(df),
        'fraud_count': df['Class'].sum(),
        'normal_count': len(df) - df['Class'].sum(),
        'fraud_percentage': (df['Class'].sum() / len(df)) * 100,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return stats

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and duplicates.
    
    Args:
        df: Raw credit card fraud dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # Remove duplicates if any
    if df_clean.duplicated().sum() > 0:
        df_clean = df_clean.drop_duplicates()
    
    # Handle missing values (though this dataset is typically clean)
    if df_clean.isnull().sum().sum() > 0:
        # For Amount column, fill with median
        if df_clean['Amount'].isnull().sum() > 0:
            df_clean['Amount'].fillna(df_clean['Amount'].median(), inplace=True)
        
        # For V columns, fill with 0 (since they're PCA transformed)
        v_columns = [f'V{i}' for i in range(1, 29)]
        for col in v_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(0, inplace=True)
    
    return df_clean

def prepare_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare basic features for modeling.
    
    Args:
        df: Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with prepared features
    """
    df_prep = df.copy()
    
    # Scale Amount feature (Time and V features are already scaled)
    scaler = RobustScaler()
    df_prep['Amount_scaled'] = scaler.fit_transform(df_prep[['Amount']])
    
    # Create time-based features
    df_prep['Hour'] = (df_prep['Time'] / 3600) % 24
    df_prep['Day'] = df_prep['Time'] // (24 * 3600)
    
    return df_prep

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers using IQR or z-score method.
    
    Args:
        df: Dataset
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        
    Returns:
        pd.Series: Boolean mask for outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def cap_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Cap outliers to reasonable bounds.
    
    Args:
        df: Dataset
        column: Column to cap outliers
        method: 'iqr' or 'zscore'
        
    Returns:
        pd.DataFrame: Dataset with capped outliers
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        
        df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df_capped

def create_amount_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Create binned features for Amount column.
    
    Args:
        df: Dataset
        n_bins: Number of bins for Amount
        
    Returns:
        pd.DataFrame: Dataset with binned Amount features
    """
    df_binned = df.copy()
    
    # Create amount bins with quantile-based binning
    df_binned['Amount_bin'] = pd.qcut(df_binned['Amount'], q=n_bins, labels=False, duplicates='drop')
    
    # Create high-value transaction indicator
    amount_95th = df_binned['Amount'].quantile(0.95)
    df_binned['High_Amount'] = (df_binned['Amount'] > amount_95th).astype(int)
    
    return df_binned

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced time-based features.
    
    Args:
        df: Dataset with Time column
        
    Returns:
        pd.DataFrame: Dataset with time features
    """
    df_time = df.copy()
    
    # Convert time to hours and days
    df_time['Hour'] = (df_time['Time'] / 3600) % 24
    df_time['Day'] = df_time['Time'] // (24 * 3600)
    
    # Create time period indicators
    df_time['Is_Weekend'] = (df_time['Day'] % 7 >= 5).astype(int)
    df_time['Is_Night'] = ((df_time['Hour'] >= 22) | (df_time['Hour'] <= 6)).astype(int)
    df_time['Is_Business_Hours'] = ((df_time['Hour'] >= 9) & (df_time['Hour'] <= 17)).astype(int)
    
    return df_time

def create_transaction_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction velocity features per user (if possible).
    Note: This dataset doesn't have user IDs, so we'll create aggregate features.
    
    Args:
        df: Dataset
        
    Returns:
        pd.DataFrame: Dataset with velocity features
    """
    df_velocity = df.copy()
    
    # Sort by time for rolling calculations
    df_velocity = df_velocity.sort_values('Time').reset_index(drop=True)
    
    # Create rolling features (simulate user transaction patterns)
    window_size = 100  # transactions
    df_velocity['Rolling_Amount_Mean'] = df_velocity['Amount'].rolling(window=window_size, min_periods=1).mean()
    df_velocity['Rolling_Amount_Std'] = df_velocity['Amount'].rolling(window=window_size, min_periods=1).std()
    df_velocity['Amount_vs_Rolling_Mean'] = df_velocity['Amount'] / (df_velocity['Rolling_Amount_Mean'] + 1e-8)
    
    # Transaction frequency in time windows
    df_velocity['Time_Since_Last'] = df_velocity['Time'].diff().fillna(0)
    df_velocity['Freq_Last_Hour'] = df_velocity['Time'].rolling(window=50, min_periods=1).apply(
        lambda x: sum((x.max() - x) <= 3600), raw=True
    )
    
    return df_velocity

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Clean dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df_engineered = df.copy()
    
    # Cap outliers in Amount
    df_engineered = cap_outliers(df_engineered, 'Amount', method='iqr')
    
    # Create time features
    df_engineered = create_time_features(df_engineered)
    
    # Create amount bins
    df_engineered = create_amount_bins(df_engineered)
    
    # Create velocity features
    df_engineered = create_transaction_velocity_features(df_engineered)
    
    # Scale Amount (keep original for interpretability)
    scaler = RobustScaler()
    df_engineered['Amount_scaled'] = scaler.fit_transform(df_engineered[['Amount']])
    
    # Create V feature interactions (top fraud indicators)
    df_engineered['V1_V2_interaction'] = df_engineered['V1'] * df_engineered['V2']
    df_engineered['V3_V4_interaction'] = df_engineered['V3'] * df_engineered['V4']
    df_engineered['V_magnitude'] = np.sqrt(df_engineered[[f'V{i}' for i in range(1, 29)]].pow(2).sum(axis=1))
    
    return df_engineered

def get_processed_data(file_path: str = 'data/creditcard.csv') -> tuple:
    """
    Main function to load, clean, and prepare the dataset with full feature engineering.
    
    Args:
        file_path: Path to the dataset
        
    Returns:
        tuple: (processed_dataframe, dataset_stats)
    """
    # Load and validate
    df = load_and_validate_data(file_path)
    
    # Explore dataset
    stats = explore_dataset(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_processed = engineer_features(df_clean)
    
    return df_processed, stats