import os
import sys

# Ensure root directory and src/ directory are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    from src.predict import CarPricePredictor, USD_TO_INR
except ImportError:
    from predict import CarPricePredictor, USD_TO_INR

# Configure Page Layout
st.set_page_config(
    page_title="Used Car Price Prediction App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }
    .price-box {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .price-main {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .price-secondary {
        font-size: 1.6rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .range-text {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    return CarPricePredictor(model_path=model_path)


@st.cache_data
def load_dataset_values():
    data_path = os.path.join(BASE_DIR, 'data', 'used_cars.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    return None


def main():
    st.markdown('<div class="main-header">🚗 Used Car Price Prediction ML App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict accurate market prices based on car characteristics & machine learning</div>', unsafe_allow_html=True)
    
    # Load dataset for dynamic selection lists
    df_raw = load_dataset_values()
    
    if df_raw is not None:
        brands = sorted(df_raw['brand'].dropna().unique())
        fuel_types = sorted([f for f in df_raw['fuel_type'].dropna().unique() if f not in ['–', '']])
        if 'Gasoline' not in fuel_types: fuel_types.append('Gasoline')
        
        transmissions = sorted([t for t in df_raw['transmission'].dropna().unique() if t not in ['–', '']])
        if 'Automatic' not in transmissions: transmissions.append('Automatic')
    else:
        brands = ['Ford', 'Hyundai', 'Lexus', 'BMW', 'Audi', 'Mercedes-Benz', 'Toyota', 'Nissan', 'Porsche', 'Tesla']
        fuel_types = ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Plug-In Hybrid', 'Diesel', 'Electric']
        transmissions = ['Automatic', '6-Speed A/T', '8-Speed Automatic', '6-Speed M/T', 'CVT Transmission', 'Transmission w/Dual Shift Mode']

    # Layout columns
    col_input, col_result = st.columns([1.2, 1.0], gap="large")
    
    with col_input:
        st.subheader("📋 Enter Vehicle Details")
        
        c1, c2 = st.columns(2)
        with c1:
            selected_brand = st.selectbox("Car Brand", options=brands, index=0)
            
            # Filter models based on brand if dataset available
            if df_raw is not None and selected_brand in df_raw['brand'].values:
                brand_models = sorted(df_raw[df_raw['brand'] == selected_brand]['model'].dropna().unique())
            else:
                brand_models = ["Base Model"]
                
            selected_model = st.selectbox("Car Model", options=brand_models, index=0)
            model_year = st.slider("Manufacturing Year", min_value=1990, max_value=2024, value=2018, step=1)
            milage = st.number_input("Mileage (in miles)", min_value=100, max_value=500000, value=45000, step=1000)

        with c2:
            fuel_type = st.selectbox("Fuel Type", options=fuel_types, index=0)
            transmission = st.selectbox("Transmission", options=transmissions, index=0)
            hp = st.number_input("Horsepower (HP)", min_value=50, max_value=1000, value=300, step=10)
            engine_liter = st.number_input("Engine Capacity (Liters)", min_value=0.8, max_value=8.4, value=3.0, step=0.1)

        c3, c4 = st.columns(2)
        with c3:
            accident = st.selectbox("Accident / Damage History", options=["None reported", "At least 1 accident or damage reported"], index=0)
        with c4:
            clean_title = st.selectbox("Clean Title Status", options=["Yes", "No"], index=0)
            
        predict_button = st.button("🔮 Predict Car Price", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Price Prediction & Analysis")
        
        if predict_button:
            try:
                predictor = load_predictor()
                
                # Format engine string for feature extractor
                engine_str = f"{hp:.1f}HP {engine_liter:.1f}L Engine"
                milage_str = f"{milage:,} mi."
                
                input_df = pd.DataFrame([{
                    'brand': selected_brand,
                    'model': selected_model,
                    'model_year': model_year,
                    'milage': milage_str,
                    'fuel_type': fuel_type,
                    'engine': engine_str,
                    'transmission': transmission,
                    'ext_col': 'Unknown',
                    'int_col': 'Unknown',
                    'accident': accident,
                    'clean_title': clean_title
                }])
                
                res = predictor.predict(input_df)
                
                pred_usd = res['predicted_price_usd']
                pred_inr = res['predicted_price_inr']
                low_usd, high_usd = res['price_range_low_usd'], res['price_range_high_usd']
                low_inr, high_inr = res['price_range_low_inr'], res['price_range_high_inr']
                model_name = res['model_name']
                
                st.markdown(f"""
                <div class="price-box">
                    <div>ESTIMATED MARKET VALUE</div>
                    <div class="price-main">${pred_usd:,.2f}</div>
                    <div class="price-secondary">₹{pred_inr:,.0f}</div>
                    <div class="range-text">
                        Estimated Range: <b>${low_usd:,.0f} - ${high_usd:,.0f}</b> (₹{low_inr:,.0f} - ₹{high_inr:,.0f})
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🔍 Summary of Inputs & Model")
                st.markdown(f"""
                - **Selected Vehicle:** {model_year} {selected_brand} {selected_model}
                - **Mileage:** {milage:,} miles (~{milage / max(1, 2024 - model_year):,.0f} mi/yr)
                - **Engine Specifications:** {hp} HP | {engine_liter} L Engine
                - **Fuel & Transmission:** {fuel_type} | {transmission}
                - **History:** Accident: *{accident}* | Clean Title: *{clean_title}*
                - **Active ML Model:** `{model_name}` (Loaded from `models/model.pkl`)
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.info("👈 Enter the vehicle specifications on the left panel and click **Predict Car Price** to get an estimate.")
            
            if df_raw is not None:
                st.markdown("### 📈 Quick Dataset Overview")
                st.metric("Total Dataset Records", f"{len(df_raw)}")
                st.metric("Average Price in Dataset", f"${df_raw['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float).mean():,.0f}")

    # Tabs at the bottom
    st.divider()
    tab1, tab2 = st.tabs(["ℹ️ About the Model", "📊 Dataset Summary"])
    
    with tab1:
        st.markdown("""
        #### Machine Learning Architecture:
        - **Pipeline:** Preprocessing includes imputation, scaling for numerical features, and One-Hot Encoding for categorical features.
        - **Models Evaluated:** Linear Regression, Random Forest Regressor, and XGBoost Regressor.
        - **Model Persistence:** The complete preprocessing + model pipeline is trained and serialized into `models/model.pkl` via `joblib`.
        """)
        
    with tab2:
        if df_raw is not None:
            st.dataframe(df_raw.head(10), use_container_width=True)


if __name__ == '__main__':
    main()
