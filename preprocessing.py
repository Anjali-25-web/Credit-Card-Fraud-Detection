import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df, is_training=True, scaler=None):
    """
    Cleans and preprocesses the Credit Card Fraud dataset.
    
    Args:
        df (pd.DataFrame): The raw input transaction data.
        is_training (bool): If True, fits the scaler and removes duplicates.
                            If False, transforms using the provided scaler and keeps all rows.
        scaler (RobustScaler, optional): Pre-fitted RobustScaler for scaling 'Time' and 'Amount'.
        
    Returns:
        If is_training=True:
            (pd.DataFrame, RobustScaler): Preprocessed DataFrame and the fitted scaler.
        If is_training=False:
            pd.DataFrame: Preprocessed DataFrame.
    """
    # Create a copy to prevent SettingWithCopyWarning
    df_clean = df.copy()
    
    # 1. Data Validation
    required_cols = ['Time', 'Amount']
    if is_training:
        required_cols.append('Class')
        
    # Check that required columns are present
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
        
    # Verify all columns (V1-V28, Time, Amount) are numeric
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    for col in feature_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    # 2. Handling Missing Values
    # Check for missing values and drop or impute
    num_missing = df_clean[feature_cols].isnull().sum().sum()
    if num_missing > 0:
        print(f"Warning: Found {num_missing} missing values. Imputing with column median...")
        # For training, we can drop rows where target is missing, and impute others
        if is_training and 'Class' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['Class'])
        # Impute features with median
        for col in feature_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                
    # 3. Duplicate Removal (Only during training to prevent training on redundant samples)
    if is_training:
        num_duplicates = df_clean.duplicated().sum()
        if num_duplicates > 0:
            print(f"Removing {num_duplicates} duplicate records...")
            df_clean = df_clean.drop_duplicates()
            
    # 4. Feature Scaling (Scaling 'Time' and 'Amount' using RobustScaler)
    # RobustScaler is chosen because 'Amount' has extreme outliers
    if is_training:
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(df_clean[['Time', 'Amount']])
        df_clean[['Time', 'Amount']] = scaled_features
        return df_clean, scaler
    else:
        if scaler is None:
            raise ValueError("A pre-fitted RobustScaler must be provided for preprocessing during inference/testing.")
        scaled_features = scaler.transform(df_clean[['Time', 'Amount']])
        df_clean[['Time', 'Amount']] = scaled_features
        return df_clean

def apply_smote(X, y, random_state=42):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution.
    This should ONLY be run on the training split to avoid data leakage.
    
    Args:
        X (pd.DataFrame or np.ndarray): Training features.
        y (pd.Series or np.ndarray): Training labels.
        random_state (int): Seed for reproducibility.
        
    Returns:
        (pd.DataFrame, pd.Series): Resampled features and labels.
    """
    print(f"Class distribution before SMOTE: {pd.Series(y).value_counts().to_dict()}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled
