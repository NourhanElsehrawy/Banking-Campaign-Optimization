
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic-based features for banking analysis
    
    Args:
        df: DataFrame with demographic columns
        
    Returns:
        pd.DataFrame: DataFrame with new demographic features
    """
    df_features = df.copy()
    
    # Age groups
    if 'age' in df_features.columns:
        age_bins = [0, 25, 35, 45, 55, 65, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df_features['age_group'] = pd.cut(df_features['age'], bins=age_bins, labels=age_labels)
    
    # Professional job indicator
    if 'job' in df_features.columns:
        professional_jobs = ['management', 'admin.', 'technician']
        df_features['is_professional'] = df_features['job'].isin(professional_jobs).astype(int)
    
    # Education level (ordinal)
    if 'education' in df_features.columns:
        education_order = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                          'professional.course', 'university.degree', 'unknown']
        df_features['education_level'] = df_features['education'].map(
            {edu: i for i, edu in enumerate(education_order)}
        ).fillna(-1).astype(int)
    
    return df_features

def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial behavior features
    
    Args:
        df: DataFrame with financial columns
        
    Returns:
        pd.DataFrame: DataFrame with new financial features
    """
    df_features = df.copy()
    
    # Balance categories
    if 'balance' in df_features.columns:
        balance_bins = [-np.inf, 0, 1000, 5000, 15000, np.inf]
        balance_labels = ['Negative', 'Low', 'Medium', 'High', 'Very High']
        df_features['balance_category'] = pd.cut(df_features['balance'], 
                                               bins=balance_bins, labels=balance_labels)
        
        # Balance per age (financial maturity)
        df_features['balance_per_age'] = df_features['balance'] / df_features['age']
    
    # Credit risk composite indicator
    risk_factors = []
    if 'default' in df_features.columns:
        risk_factors.append(df_features['default'] == 'yes')
    if 'loan' in df_features.columns:
        risk_factors.append(df_features['loan'] == 'yes')
    
    if risk_factors:
        df_features['has_credit_risk'] = np.any(risk_factors, axis=0).astype(int)
    
    return df_features

def create_campaign_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create campaign interaction features
    
    Args:
        df: DataFrame with campaign columns
        
    Returns:
        pd.DataFrame: DataFrame with new campaign features
    """
    df_features = df.copy()
    
    # Call efficiency
    if 'duration' in df_features.columns and 'campaign' in df_features.columns:
        df_features['call_efficiency'] = df_features['duration'] / df_features['campaign']
    
    # Previous contact indicators
    if 'pdays' in df_features.columns:
        df_features['was_contacted_before'] = (df_features['pdays'] != -1).astype(int)
        
        # Contact recency categories
        df_features['contact_recency'] = pd.cut(
            df_features['pdays'], 
            bins=[-2, -1, 7, 30, 180, 999], 
            labels=['never', 'recent', 'medium', 'old', 'very_old']
        )
    
    # Previous campaign success
    if 'poutcome' in df_features.columns:
        df_features['prev_campaign_success'] = (df_features['poutcome'] == 'success').astype(int)
    
    # Contact intensity
    if 'campaign' in df_features.columns:
        median_campaigns = df_features['campaign'].median()
        df_features['high_contact_intensity'] = (df_features['campaign'] > median_campaigns).astype(int)
    
    return df_features

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal/seasonal features
    
    Args:
        df: DataFrame with temporal columns
        
    Returns:
        pd.DataFrame: DataFrame with new temporal features
    """
    df_features = df.copy()
    
    if 'month' in df_features.columns:
        # Month numeric mapping
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        df_features['month_numeric'] = df_features['month'].map(month_map)
        
        # Seasonal indicators
        df_features['is_q1'] = df_features['month_numeric'].isin([1, 2, 3]).astype(int)
        df_features['is_summer'] = df_features['month_numeric'].isin([6, 7, 8]).astype(int)
        df_features['is_q4'] = df_features['month_numeric'].isin([10, 11, 12]).astype(int)
    
    return df_features

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced interaction features
    
    Args:
        df: DataFrame with base features
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    df_features = df.copy()
    
    # Age-education interaction (mature + educated = high value)
    if 'age' in df_features.columns and 'education_level' in df_features.columns:
        df_features['mature_educated'] = (
            (df_features['age'] > 35) & (df_features['education_level'] >= 5)
        ).astype(int)
    
    # Duration-previous success interaction
    if 'duration' in df_features.columns and 'prev_campaign_success' in df_features.columns:
        df_features['duration_prev_success'] = (
            df_features['duration'] * df_features['prev_campaign_success']
        )
    
    return df_features

def encode_categorical_features(df: pd.DataFrame, 
                               exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features for machine learning
    
    Args:
        df: DataFrame with categorical features
        exclude_cols: Columns to exclude from encoding
        
    Returns:
        tuple: (encoded_df, dict_of_encoders)
    """
    if exclude_cols is None:
        exclude_cols = ['y', 'target']
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Get categorical columns for encoding
    categorical_cols = df_encoded.select_dtypes(include=['category', 'object']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_col = f'{col}_encoded'
        df_encoded[encoded_col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to appropriate data types
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    df_optimized = df.copy()
    
    # Convert appropriate columns to categorical
    categorical_candidates = ['job', 'marital', 'education', 'default', 'housing', 
                             'loan', 'contact', 'poutcome', 'month', 'day_of_week',
                             'age_group', 'balance_category', 'contact_recency']
    
    for col in categorical_candidates:
        if col in df_optimized.columns and df_optimized[col].dtype == 'object':
            df_optimized[col] = df_optimized[col].astype('category')
    
    # Optimize integer types
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col in df_optimized.columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:  # Signed integers
                if col_min >= -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min >= -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min >= -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
    
    return df_optimized

def create_target_variable(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """
    Create binary target variable from categorical target
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        
    Returns:
        pd.DataFrame: DataFrame with binary target
    """
    df_target = df.copy()
    
    if target_col in df_target.columns:
        df_target['target'] = (df_target[target_col] == 'yes').astype(int)
    
    return df_target

def create_customer_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer lifetime value and segmentation features
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        pd.DataFrame: DataFrame with customer value features
    """
    df_features = df.copy()
    
    # Customer value score based on multiple factors
    if all(col in df_features.columns for col in ['age', 'balance', 'education_level']):
        # Normalize factors to 0-1 scale
        age_score = (df_features['age'] - 18) / (65 - 18)  # Peak earning years
        age_score = np.where(df_features['age'] > 65, 0.5, age_score)  # Retired penalty
        
        balance_score = np.clip(df_features['balance'] / 20000, 0, 1)  # Cap at 20k
        education_score = df_features['education_level'] / 6  # Max education level
        
        df_features['customer_value_score'] = (
            age_score * 0.3 + balance_score * 0.5 + education_score * 0.2
        )
    
    # High-value customer indicator
    if 'customer_value_score' in df_features.columns:
        threshold = df_features['customer_value_score'].quantile(0.75)
        df_features['is_high_value'] = (df_features['customer_value_score'] > threshold).astype(int)
    
    # Customer stability indicator
    stability_factors = []
    if 'marital' in df_features.columns:
        stability_factors.append((df_features['marital'] == 'married').astype(int))
    if 'housing' in df_features.columns:
        stability_factors.append((df_features['housing'] == 'yes').astype(int))
    if 'job' in df_features.columns:
        stable_jobs = ['management', 'admin.', 'technician', 'services']
        stability_factors.append(df_features['job'].isin(stable_jobs).astype(int))
    
    if stability_factors:
        df_features['customer_stability'] = np.mean(stability_factors, axis=0)
    
    return df_features

def create_campaign_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create campaign efficiency and optimization features
    
    Args:
        df: DataFrame with campaign data
        
    Returns:
        pd.DataFrame: DataFrame with campaign efficiency features
    """
    df_features = df.copy()
    
    # Call quality indicators
    if 'duration' in df_features.columns:
        # Duration percentiles
        df_features['duration_percentile'] = pd.qcut(
            df_features['duration'], q=10, labels=False
        ) + 1
        
        # Optimal duration indicator (based on EDA insights)
        optimal_duration_range = (300, 600)  # 5-10 minutes
        df_features['optimal_duration'] = (
            (df_features['duration'] >= optimal_duration_range[0]) & 
            (df_features['duration'] <= optimal_duration_range[1])
        ).astype(int)
    
    # Campaign fatigue indicator
    if 'campaign' in df_features.columns:
        # High campaign frequency might indicate fatigue
        campaign_threshold = df_features['campaign'].quantile(0.8)
        df_features['campaign_fatigue'] = (df_features['campaign'] > campaign_threshold).astype(int)
        
        # Campaign efficiency relative to median
        median_campaigns = df_features['campaign'].median()
        df_features['campaign_above_median'] = (df_features['campaign'] > median_campaigns).astype(int)
    
    # Contact method effectiveness
    if 'contact' in df_features.columns:
        # Based on typical effectiveness: cellular > telephone > unknown
        contact_effectiveness = {'cellular': 2, 'telephone': 1, 'unknown': 0}
        df_features['contact_effectiveness'] = df_features['contact'].map(contact_effectiveness).fillna(0)
    
    # Previous success momentum
    if 'prev_campaign_success' in df_features.columns and 'previous' in df_features.columns:
        # Success rate from previous campaigns
        df_features['prev_success_rate'] = np.where(
            df_features['previous'] > 0,
            df_features['prev_campaign_success'] / df_features['previous'],
            0
        )
    
    return df_features

def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral and psychological profiling features
    
    Args:
        df: DataFrame with behavioral indicators
        
    Returns:
        pd.DataFrame: DataFrame with behavioral features
    """
    df_features = df.copy()
    
    # Risk appetite indicator
    risk_indicators = []
    if 'loan' in df_features.columns:
        risk_indicators.append((df_features['loan'] == 'yes').astype(int))
    if 'default' in df_features.columns:
        risk_indicators.append((df_features['default'] == 'yes').astype(int))
    
    if risk_indicators:
        df_features['risk_appetite'] = np.mean(risk_indicators, axis=0)
    
    # Financial engagement level
    engagement_factors = []
    if 'housing' in df_features.columns:
        engagement_factors.append((df_features['housing'] == 'yes').astype(int))
    if 'balance' in df_features.columns:
        engagement_factors.append((df_features['balance'] > 0).astype(int))
    if 'previous' in df_features.columns:
        engagement_factors.append((df_features['previous'] > 0).astype(int))
    
    if engagement_factors:
        df_features['financial_engagement'] = np.mean(engagement_factors, axis=0)
    
    # Decision-making speed (based on call duration and success)
    if all(col in df_features.columns for col in ['duration', 'target']):
        # Quick decision makers: short duration but high success rate
        quick_threshold = df_features['duration'].quantile(0.3)
        df_features['quick_decision_maker'] = (
            (df_features['duration'] <= quick_threshold) & 
            (df_features['target'] == 1)
        ).astype(int)
    
    return df_features

def create_economic_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create economic context and market timing features
    
    Args:
        df: DataFrame with economic indicators
        
    Returns:
        pd.DataFrame: DataFrame with economic context features
    """
    df_features = df.copy()
    
    # Economic indicators (if available)
    economic_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    available_economic = [col for col in economic_cols if col in df_features.columns]
    
    if available_economic:
        # Economic optimism composite score
        if 'cons.conf.idx' in df_features.columns and 'euribor3m' in df_features.columns:
            # Higher confidence = good, lower interest rates = good
            confidence_norm = (df_features['cons.conf.idx'] - df_features['cons.conf.idx'].min()) / \
                            (df_features['cons.conf.idx'].max() - df_features['cons.conf.idx'].min())
            interest_norm = 1 - ((df_features['euribor3m'] - df_features['euribor3m'].min()) / \
                               (df_features['euribor3m'].max() - df_features['euribor3m'].min()))
            
            df_features['economic_optimism'] = (confidence_norm + interest_norm) / 2
        
        # Market timing indicators
        if 'euribor3m' in df_features.columns:
            low_interest_threshold = df_features['euribor3m'].quantile(0.25)
            df_features['low_interest_period'] = (
                df_features['euribor3m'] <= low_interest_threshold
            ).astype(int)
    
    # Seasonal economic effects
    if 'month_numeric' in df_features.columns:
        # End of year financial planning season
        df_features['financial_planning_season'] = df_features['month_numeric'].isin([10, 11, 12]).astype(int)
        
        # Tax season effects
        df_features['tax_season'] = df_features['month_numeric'].isin([3, 4]).astype(int)
        
        # Summer vacation impact
        df_features['vacation_season'] = df_features['month_numeric'].isin([7, 8]).astype(int)
    
    return df_features

def create_advanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create complex interaction features for advanced modeling
    
    Args:
        df: DataFrame with base features
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    df_features = df.copy()
    
    # Age-Balance interaction (wealth accumulation patterns)
    if all(col in df_features.columns for col in ['age', 'balance']):
        # Expected balance for age (simple linear model)
        expected_balance = (df_features['age'] - 18) * 1000  # $1k per year after 18
        df_features['balance_vs_expected'] = df_features['balance'] / (expected_balance + 1)
        df_features['above_expected_wealth'] = (df_features['balance_vs_expected'] > 1.5).astype(int)
    
    # Education-Job alignment
    if all(col in df_features.columns for col in ['education_level', 'job']):
        high_ed_prof_jobs = ['management', 'admin.', 'technician']
        df_features['education_job_alignment'] = (
            (df_features['education_level'] >= 5) & 
            (df_features['job'].isin(high_ed_prof_jobs))
        ).astype(int)
    
    # Campaign strategy effectiveness
    if all(col in df_features.columns for col in ['duration', 'campaign', 'contact_effectiveness']):
        # Quality vs quantity strategy
        df_features['quality_strategy'] = (
            (df_features['duration'] > df_features['duration'].median()) & 
            (df_features['campaign'] <= df_features['campaign'].median()) &
            (df_features['contact_effectiveness'] >= 1)
        ).astype(int)
    
    # Life stage indicators
    if all(col in df_features.columns for col in ['age', 'marital', 'housing']):
        # Young professional
        df_features['young_professional'] = (
            (df_features['age'].between(25, 40)) &
            (df_features['marital'].isin(['single', 'married'])) &
            (df_features['is_professional'] == 1)
        ).astype(int)
        
        # Established family
        df_features['established_family'] = (
            (df_features['age'].between(35, 55)) &
            (df_features['marital'] == 'married') &
            (df_features['housing'] == 'yes')
        ).astype(int)
    
    return df_features

def select_model_features(df: pd.DataFrame, 
                         feature_importance_threshold: float = 0.01,
                         correlation_threshold: float = 0.95) -> List[str]:
    """
    Intelligent feature selection based on importance and correlation
    
    Args:
        df: DataFrame with all features
        feature_importance_threshold: Minimum importance to keep feature
        correlation_threshold: Maximum correlation to keep both features
        
    Returns:
        list: Selected feature names for modeling
    """
    # Exclude non-modeling columns
    exclude_cols = ['y', 'target'] + [col for col in df.columns if col.endswith('_group') or col.endswith('_category')]
    
    # Start with numerical and encoded features
    potential_features = []
    for col in df.columns:
        if col not in exclude_cols and col in df.columns:
            # Include numerical features
            if df[col].dtype in ['int64', 'float64', 'int32', 'int16', 'int8']:
                potential_features.append(col)
            # Include encoded categorical features
            elif col.endswith('_encoded'):
                potential_features.append(col)
            # Include binary indicators
            elif df[col].nunique() == 2 and df[col].dtype in ['int64', 'int8']:
                potential_features.append(col)
    
    # Remove highly correlated features
    if len(potential_features) > 1:
        corr_matrix = df[potential_features].corr().abs()
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname_i = corr_matrix.columns[i]
                    colname_j = corr_matrix.columns[j]
                    high_corr_pairs.append((colname_i, colname_j, corr_matrix.iloc[i, j]))
        
        # Remove features from highly correlated pairs (keep first alphabetically)
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            # Remove the lexicographically later feature
            if feat1 > feat2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        potential_features = [f for f in potential_features if f not in features_to_remove]
    
    return potential_features

def prepare_features_for_scaling(df: pd.DataFrame, feature_list: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features for appropriate scaling strategies
    
    Args:
        df: DataFrame with features
        feature_list: List of features to categorize
        
    Returns:
        dict: Categories of features for different scaling approaches
    """
    feature_categories = {
        'binary_features': [],
        'count_features': [],
        'ratio_features': [],
        'continuous_features': [],
        'encoded_features': []
    }
    
    for feature in feature_list:
        if feature in df.columns:
            # Binary features (0/1)
            if df[feature].nunique() == 2 and set(df[feature].unique()).issubset({0, 1}):
                feature_categories['binary_features'].append(feature)
            
            # Encoded categorical features
            elif feature.endswith('_encoded'):
                feature_categories['encoded_features'].append(feature)
            
            # Count features (integers, likely counts)
            elif df[feature].dtype in ['int64', 'int32'] and df[feature].min() >= 0:
                if 'campaign' in feature or 'previous' in feature or 'contact' in feature:
                    feature_categories['count_features'].append(feature)
                else:
                    feature_categories['continuous_features'].append(feature)
            
            # Ratio features (calculated ratios)
            elif any(keyword in feature for keyword in ['_per_', 'efficiency', 'rate', 'ratio']):
                feature_categories['ratio_features'].append(feature)
            
            # Continuous features (default)
            else:
                feature_categories['continuous_features'].append(feature)
    
    return feature_categories

def create_feature_engineering_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete feature engineering pipeline
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        tuple: (feature_engineered_df, metadata)
    """

       # Step 6: Create target variable
    df_clean = create_target_variable(df)
    
    # Step 7: Feature engineering
    df_clean = create_demographic_features(df_clean)
    df_clean = create_financial_features(df_clean)
    df_clean = create_campaign_features(df_clean)
    df_clean = create_temporal_features(df_clean)
    df_clean = create_interaction_features(df_clean)
       
    # Apply all feature engineering functions
    df_features = df_clean.copy()
    
    # Customer value features
    df_features = create_customer_value_features(df_features)
   
    # Campaign efficiency features
    df_features = create_campaign_efficiency_features(df_features)
  
    # Behavioral features
    df_features = create_behavioral_features(df_features)
   
    # Economic context features
    df_features = create_economic_context_features(df_features)

    # Advanced interactions
    df_features = create_advanced_interaction_features(df_features)
    
    # Feature selection
    model_features = select_model_features(df_features)

   
    
    # Final feature count
    print(f"final_features:{ list(df_features.columns)}")
    print(f"features_added: {len(df_features.columns) - len(df.columns)}")
    
    return df_features