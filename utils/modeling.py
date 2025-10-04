from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
# System & utils
import sys, os
notebook_dir = os.path.dirname(os.getcwd())
sys.path.append(notebook_dir)

# Data
import pandas as pd
import numpy as np



def encode_categorical_features(df, categorical_cols=None, method="label"):
    df_encoded = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == "label":

        oe = OrdinalEncoder()
        df_encoded[categorical_cols] = oe.fit_transform(df_encoded[categorical_cols])
    elif method == "onehot":
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    else:
        raise ValueError("method must be 'label' or 'onehot'")
    
    return df_encoded

def scale_data(X_train, X_test, apply_scaling=True):
    scaler = StandardScaler()
    if apply_scaling:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:] = scaler.fit_transform(X_train)
        X_test_scaled[:] = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    return X_train_scaled, X_test_scaled, scaler
