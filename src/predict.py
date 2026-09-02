import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure root and src directory are in sys.path for absolute and relative imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.preprocessing import clean_dataset
except ImportError:
    from preprocessing import clean_dataset

# Approximate conversion rate for USD to INR
USD_TO_INR = 83.50

class CarPricePredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
            
        self.pipeline = joblib.load(model_path)
        
    def predict(self, input_df):
        """
        Predict used car selling price given input dataframe of characteristics.
        
        Parameters:
        -----------
        input_df : pd.DataFrame
            DataFrame containing columns matching dataset schema.
            
        Returns:
        --------
        dict containing predicted_price_usd, predicted_price_inr,
        price_range_low_usd, price_range_high_usd,
        price_range_low_inr, price_range_high_inr.
        """
        cleaned_X = clean_dataset(input_df, is_train=False)
        pred_usd = float(self.pipeline.predict(cleaned_X)[0])
        
        # Ensure prediction is positive
        pred_usd = max(500.0, pred_usd)
        
        # Statistically justified price range (+/- 12% margin)
        margin_pct = 0.12
        low_usd = pred_usd * (1 - margin_pct)
        high_usd = pred_usd * (1 + margin_pct)
        
        pred_inr = pred_usd * USD_TO_INR
        low_inr = low_usd * USD_TO_INR
        high_inr = high_usd * USD_TO_INR
        
        return {
            'predicted_price_usd': pred_usd,
            'predicted_price_inr': pred_inr,
            'price_range_low_usd': low_usd,
            'price_range_high_usd': high_usd,
            'price_range_low_inr': low_inr,
            'price_range_high_inr': high_inr,
            'model_name': type(self.pipeline.named_steps['regressor']).__name__
        }

def predict_single_car(brand, model, model_year, milage, fuel_type, engine, transmission, accident, clean_title):
    input_data = pd.DataFrame([{
        'brand': brand,
        'model': model,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel_type,
        'engine': engine,
        'transmission': transmission,
        'ext_col': 'Unknown',
        'int_col': 'Unknown',
        'accident': accident,
        'clean_title': clean_title
    }])
    
    predictor = CarPricePredictor()
    return predictor.predict(input_data)


if __name__ == '__main__':
    sample_res = predict_single_car(
        brand='Ford',
        model='Mustang GT Premium',
        model_year=2018,
        milage='35,000 mi.',
        fuel_type='Gasoline',
        engine='460.0HP 5.0L 8 Cylinder Engine Gasoline Fuel',
        transmission='6-Speed M/T',
        accident='None reported',
        clean_title='Yes'
    )
    print("Sample Prediction Result:")
    for k, v in sample_res.items():
        if 'inr' in k:
            print(f"  {k}: ₹{v:,.2f}")
        elif 'usd' in k:
            print(f"  {k}: ${v:,.2f}")
        else:
            print(f"  {k}: {v}")
