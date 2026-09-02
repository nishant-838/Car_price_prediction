import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Ensure root and src directory are in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.preprocessing import clean_dataset, build_preprocessor
except ImportError:
    from preprocessing import clean_dataset, build_preprocessor


def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"--- 1. LOADING DATASET FROM: {data_path} ---")
    df = pd.read_csv(data_path)
    
    # Clean dataset and extract features
    X, y = clean_dataset(df, is_train=True)
    
    # Filter valid target values
    y_float = pd.to_numeric(y, errors='coerce').astype(float)
    valid_mask = y_float.notna() & (y_float > 0)
    X = X[valid_mask].reset_index(drop=True)
    y = y_float[valid_mask].reset_index(drop=True)
    
    print(f"Cleaned dataset shape: X = {X.shape}, y = {y.shape}")
    
    # Define features
    numerical_features = ['model_year', 'milage', 'hp', 'engine_liter', 'car_age', 'milage_per_year']
    categorical_features = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    # Define base models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    }
    
    print("\n--- 2. CROSS-VALIDATION ON TRAIN SET (5-Fold) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        neg_mse_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        cv_results[name] = {
            'CV_RMSE_mean': rmse_scores.mean(),
            'CV_RMSE_std': rmse_scores.std(),
            'CV_R2_mean': r2_scores.mean()
        }
        print(f"{name}: CV RMSE = ${rmse_scores.mean():,.2f} (+/- ${rmse_scores.std():,.2f}), CV R2 = {r2_scores.mean():.4f}")
        
    print("\n--- 3. HYPERPARAMETER TUNING FOR RANDOM FOREST & XGBOOST ---")
    
    # Random Forest Tuning
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    rf_param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
    
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {rf_grid.best_params_}")
    best_rf_pipeline = rf_grid.best_estimator_
    
    # XGBoost Tuning
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    
    xgb_param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 6],
        'regressor__learning_rate': [0.05, 0.1]
    }
    
    xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {xgb_grid.best_params_}")
    best_xgb_pipeline = xgb_grid.best_estimator_
    
    # Evaluate All Models on Test Set
    lr_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    
    eval_models = {
        'Linear Regression': lr_pipeline,
        'Random Forest (Tuned)': best_rf_pipeline,
        'XGBoost (Tuned)': best_xgb_pipeline
    }
    
    print("\n--- 4. FINAL TEST SET EVALUATION TABLE ---")
    results_list = []
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None
    
    for name, pipe in eval_models.items():
        preds = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        results_list.append({
            'Model': name,
            'MAE ($)': round(mae, 2),
            'RMSE ($)': round(rmse, 2),
            'R² Score': round(r2, 4)
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipe

    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    
    print(f"\n>>> BEST PERFORMING MODEL: {best_model_name} with R² = {best_r2:.4f} <<<")
    
    # Save the complete pipeline
    model_save_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best pipeline model to: {model_save_path}")
    
    # Feature Importances Analysis
    regressor = best_pipeline.named_steps['regressor']
    preproc = best_pipeline.named_steps['preprocessor']
    
    if hasattr(regressor, 'feature_importances_'):
        print("\n--- 5. FEATURE IMPORTANCE ANALYSIS ---")
        try:
            cat_encoder = preproc.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
            all_feature_names = numerical_features + cat_feature_names
            importances = regressor.feature_importances_
            
            fi_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print(fi_df.head(15).to_string(index=False))
        except Exception as e:
            print(f"Feature importance detail error: {e}")


if __name__ == '__main__':
    train_and_evaluate()
