import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality assessment
    
    Args:
        df: DataFrame to assess
        
    Returns:
        dict: Quality assessment report
    """
    assessment = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    # Outlier detection for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numerical_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = (z_scores > 3).sum()
    
    assessment['outliers'] = outliers

        
    # Print the top-level information
    print("--- Dataset Overview ---")
    print(f"Shape: {assessment['shape']}")
    print(f"Memory Usage: {assessment['memory_usage_mb']:.2f} MB")
    print(f"Duplicate Rows: {assessment['duplicate_rows']}")
    
    # Print the missing values
    print("\n--- Missing Values ---")
    for column, count in assessment['missing_values'].items():
        print(f"- {column.capitalize()}: {count} missing values")
    
    # Print the data types
    print("\n--- Data Types ---")
    for dtype, count in assessment['data_types'].items():
        print(f"- {dtype}: {count} columns")
    
    # Print the outliers
    print("\n--- Outlier Counts ---")
    for column, count in assessment['outliers'].items():
        
      
        print(f"- {column.capitalize()}: {count} outliers")
    numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

    # Plotting the box plots in a 3-column, 2-row grid
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Box Plot of {col.capitalize()}')
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel(col.capitalize())
        
    plt.tight_layout()
    plt.show()
    
def validate_business_logic(df: pd.DataFrame) -> List[str]:
    """
    Validate business-specific rules for banking data
    
    Args:
        df: DataFrame to validate
        
    Returns:
        list: List of validation issues found
    """
    issues = []
    
    # Age validation
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 18) | (df['age'] > 105)).sum()
        if invalid_age > 0:
            issues.append(f"Invalid ages: {invalid_age} records outside 18-100 range")
    
    # Duration validation
    if 'duration' in df.columns:
        negative_duration = (df['duration'] < 0).sum()
        if negative_duration > 0:
            issues.append(f"Negative durations: {negative_duration} records")
    
    # Previous contact logic
    if 'previous' in df.columns and 'pdays' in df.columns:
        logic_violations = ((df['previous'] == 0) & (df['pdays'] != -1)).sum()
        logic_violations += ((df['previous'] > 0) & (df['pdays'] == -1)).sum()
        if logic_violations > 0:
            issues.append(f"Previous contact logic violations: {logic_violations} records")
    
    # Balance validation
    if 'balance' in df.columns:
        extreme_balances = ((df['balance'] < -100000) | (df['balance'] > 1000000)).sum()
        if extreme_balances > 0:
            issues.append(f"Extreme balance values: {extreme_balances} records")
    
    return issues

def handle_outliers(df: pd.DataFrame, method: str = 'clip') -> Tuple[pd.DataFrame, List[str]]:
    """
    Handle outliers using business-informed approach
    
    Args:
        df: DataFrame to process
        method: Method to handle outliers ('clip', 'remove', 'transform')
        
    Returns:
        tuple: (cleaned_df, list_of_actions_taken)
    """
    df_clean = df.copy()
    actions = []
    
    # Duration: Cap at 2 hours (reasonable for banking calls)
    if 'duration' in df_clean.columns:
        max_duration = 7200  # 2 hours in seconds
        outliers_count = (df_clean['duration'] > max_duration).sum()
        if outliers_count > 0:
            df_clean['duration'] = df_clean['duration'].clip(upper=max_duration)
            actions.append(f"Capped {outliers_count} duration outliers at {max_duration}s")
    

    
    # Campaign: Cap at reasonable maximum
    if 'campaign' in df_clean.columns:
        max_campaigns = 40
        outliers_count = (df_clean['campaign'] > max_campaigns).sum()
        if outliers_count > 0:
            df_clean['campaign'] = df_clean['campaign'].clip(upper=max_campaigns)
            actions.append(f"Capped {outliers_count} campaign outliers at {max_campaigns}")
    
    return df_clean, actions

def standardize_categorical(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Standardize categorical variables (case, unknown values)
    
    Args:
        df: DataFrame to standardize
        exclude_cols: Columns to exclude from standardization
        
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
   
    df_clean = df.copy()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    for col in categorical_cols:
        # Standardize case and whitespace
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
        
        # Handle unknown/missing variants
        unknown_variants = ['unknown', 'unk', 'na', 'n/a', '', 'none', 'nan']
        df_clean[col] = df_clean[col].replace(unknown_variants, 'unknown')
    
    return df_clean



def complete_data_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data cleaning pipeline combining all functions
    
    Args:
        df: Raw DataFrame to clean
        
    Returns:
        tuple: (cleaned_df, cleaning_metadata)
    """

    
    # Step 1: Initial assessment
    initial_assessment = assess_data_quality(df)
    
    
    # Step 2: Business logic validation
    validation_issues = validate_business_logic(df)
    if validation_issues:
        print(validation_issues)
    
    
    # Step 3: Handle duplicates
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Step 4: Handle outliers
    df_clean, outlier_actions = handle_outliers(df_clean)
    print(outlier_actions)
    
    # Step 5: Standardize categorical
    df_clean = standardize_categorical(df_clean)
   

    
    return df_clean
    