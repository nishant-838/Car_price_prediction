import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def extract_hp(engine_str):
    """
    Extract numerical horsepower (HP) from the engine description string.
    Example: '300.0HP 3.7L V6 Cylinder Engine' -> 300.0
    """
    if pd.isna(engine_str):
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)\s*HP', str(engine_str), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan


def extract_engine_liter(engine_str):
    """
    Extract engine displacement in Liters from the engine description string.
    Example: '3.8L V6 24V GDI DOHC' -> 3.8
             '3.5 Liter DOHC' -> 3.5
    """
    if pd.isna(engine_str):
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)\s*L', str(engine_str), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match2 = re.search(r'(\d+(?:\.\d+)?)\s*Liter', str(engine_str), re.IGNORECASE)
    if match2:
        return float(match2.group(1))
    return np.nan


def clean_dataset(df, is_train=True):
    """
    Clean raw DataFrame and perform feature extraction / engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset input.
    is_train : bool
        If True, extracts and returns target series 'y' along with 'X'.
        
    Returns:
    --------
    X : pd.DataFrame
        Cleaned feature matrix.
    y : pd.Series (optional, if is_train=True)
        Target price values (float).
    """
    data = df.copy()
    
    # Target price cleaning if present
    y = None
    if 'price' in data.columns:
        price_clean = (
            data['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        y = pd.to_numeric(price_clean, errors='coerce').astype(float)
    
    # Clean Mileage
    if 'milage' in data.columns:
        milage_clean = (
            data['milage']
            .astype(str)
            .str.replace(' mi.', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        data['milage'] = pd.to_numeric(milage_clean, errors='coerce').astype(float)
    
    # Feature Engineering: Engine HP and Liter extraction
    if 'engine' in data.columns:
        data['hp'] = data['engine'].apply(extract_hp)
        data['engine_liter'] = data['engine'].apply(extract_engine_liter)
    else:
        if 'hp' not in data.columns:
            data['hp'] = np.nan
        if 'engine_liter' not in data.columns:
            data['engine_liter'] = np.nan
            
    # Feature Engineering: Car Age & Mileage Per Year
    current_year = 2024
    if 'model_year' in data.columns:
        data['model_year'] = pd.to_numeric(data['model_year'], errors='coerce').astype(float)
        data['car_age'] = current_year - data['model_year']
        data['car_age'] = data['car_age'].apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0)
    else:
        data['car_age'] = 0.0
        
    data['milage_per_year'] = data['milage'] / (data['car_age'] + 1.0)
    
    # Handle missing / placeholder strings in categorical columns
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown').astype(str).str.strip()
            data[col] = data[col].replace({'–': 'Unknown', '': 'Unknown'})
            
    # Drop target column from feature matrix
    feature_cols = [c for c in data.columns if c not in ['price', 'price_num']]
    X = data[feature_cols]
    
    if is_train and y is not None:
        return X, y
    return X


def build_preprocessor(numerical_features, categorical_features):
    """
    Build a Scikit-Learn ColumnTransformer for numerical and categorical features.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor
