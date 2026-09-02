# 🚗 Used Car Price Prediction ML Project & Streamlit App

An end-to-end Data Science and Machine Learning project that predicts the market selling price of used cars based on their characteristics using Python, Scikit-Learn, XGBoost, and a Streamlit web application.

---

## 📌 Project Overview
The objective of this project is to build a robust machine learning regression model to estimate used car prices accurately and deploy it as an interactive Streamlit web application. 

### Key Highlights:
- **Data-Driven & Empirical:** Uses the actual Kaggle used car dataset (`272` vehicle listings) without synthetic or fabricated data.
- **End-to-End Pipeline:** Complete workflow including data cleaning, regex-based feature extraction, feature engineering, exploratory data analysis (EDA), model benchmarking with 5-fold cross-validation, hyperparameter tuning, model interpretation, and Streamlit web deployment.
- **Modular & Production-Ready:** Uses Scikit-Learn `Pipeline` and `ColumnTransformer` serialized with `joblib` so that inference in the Streamlit app applies exact training transformations seamlessly.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit

---

## 📁 Project Structure
```text
used-car-price-prediction/
│
├── data/
│   └── used_cars.csv           # Source Kaggle used car dataset (272 records)
│
├── models/
│   └── model.pkl               # Saved scikit-learn preprocessing + model pipeline
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis notebook with visualizations
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning, regex extraction, & ColumnTransformer builder
│   ├── train.py                # Model training, CV, hyperparameter tuning, & evaluation
│   └── predict.py              # Prediction wrapper with confidence range calculation
│
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive project documentation
└── .gitignore                  # Git ignore configuration
```

---

## 📊 Dataset & Feature Overview

The dataset (`data/used_cars.csv`) contains 272 rows and 12 raw attributes:

| Feature Name | Type | Description | Cleaning / Transformation |
| :--- | :--- | :--- | :--- |
| `brand` | Categorical | Car manufacturer (Ford, BMW, Porsche, etc.) | Trimmed & missing handled |
| `model` | Categorical | Specific car model string | Dynamically linked to brand in UI |
| `model_year` | Numerical | Manufacturing year (1993–2023) | Used to compute `car_age` |
| `milage` | String -> Numerical | Recorded odometer mileage | Parsed from `"XX,XXX mi."` to `float` |
| `fuel_type` | Categorical | Gasoline, Hybrid, E85 Flex, Diesel, Electric | Missing values imputed with mode |
| `engine` | String -> Features | Raw engine text (e.g. `300.0HP 3.7L V6...`) | **Regex Extracted:** `hp` (Horsepower) & `engine_liter` (Displacement) |
| `transmission` | Categorical | Automatic, Manual, Dual Shift Mode, CVT | Normalized categories |
| `accident` | Categorical | History of reported accident or damage | Missing imputed as `"None reported"` |
| `clean_title` | Categorical | Title status (`Yes`, missing) | Missing imputed as `"No"` |
| `price` | String -> Numerical | Selling price (Target variable) | Parsed from `"$XX,XXX"` to `float` |

---

## ⚙️ Data Cleaning & Feature Engineering

### 1. Data Cleaning
- **Target (`price`):** Stripped `$`, `,` symbols and cast to numeric `float64`.
- **Mileage (`milage`):** Stripped `" mi."`, `,` symbols and cast to numeric `float64`.
- **Missing Values:** Imputed categorical missing values with domain-appropriate modes (`'Gasoline'`, `'None reported'`).

### 2. Feature Extraction
- **Horsepower (`hp`):** Extracted numerical horsepower using regex `r'(\d+(?:\.\d+)?)\s*HP'`.
- **Engine Size (`engine_liter`):** Extracted engine displacement in Liters using regex `r'(\d+(?:\.\d+)?)\s*L(?:iter)?'`.

### 3. Feature Engineering
- **Car Age (`car_age`):** `2024 - model_year` (represents vehicle age in years).
- **Mileage Per Year (`milage_per_year`):** `milage / (car_age + 1.0)` (handles zero-age vehicles cleanly to avoid division-by-zero).

---

## 🏆 Model Benchmarking & Evaluation

The dataset was split into **80% Training (217 samples)** and **20% Testing (55 samples)**. Models were evaluated using 5-Fold Cross-Validation on the training set and hyperparameter tuning via `GridSearchCV`.

### Model Evaluation Results (Test Set):

| Model Name | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | $28,339.57 | $46,364.72 | 0.4856 |
| **XGBoost (Tuned)** | $21,363.72 | $52,367.20 | 0.3438 |
| **Random Forest (Tuned)** 🏆 | **$19,247.88** | **$41,386.53** | **0.5901** |

> **Best Model:** **Random Forest Regressor (Tuned)** achieved the highest test **R² score of 0.5901** and lowest **MAE of $19,247.88**.

### Top 5 Feature Importances:
1. `engine_liter` (Engine Displacement): **30.38%**
2. `brand_Bugatti` (Supercar Brand Flag): **21.71%**
3. `milage_per_year` (Usage Intensity): **19.10%**
4. `milage` (Total Odometer Reading): **7.94%**
5. `transmission` (Transmission Type): **6.24%**

---

## 🚀 How to Run the Project Locally (Mac/Linux/Windows)

### 1. Prerequisites & Virtual Environment Setup
Open your terminal in the project directory and run:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (Mac/Linux)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Train the ML Models & Export Pipeline
To re-run data preprocessing, cross-validation, hyperparameter tuning, evaluation, and save `models/model.pkl`:

```bash
python3 src/train.py
```

### 3. Launch the Streamlit Web Application
To launch the interactive web application locally:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 💡 Interview Reasoning & Technical FAQ

- **Why use Scikit-Learn Pipelines?** Combining `ColumnTransformer` and the regressor into a single pipeline prevents data leakage during cross-validation and guarantees that inference transforms unseen inputs identically to training data.
- **How is division-by-zero avoided in Feature Engineering?** `milage_per_year` is computed as `milage / (car_age + 1.0)`, ensuring brand-new cars (age = 0) do not throw zero-division errors.
- **Why format in both USD and INR?** The original dataset records prices in USD ($). Converting predictions to INR (₹) using real-time equivalent rates offers enhanced usability for international resume portfolios.

---

## 🔮 Future Improvements
1. **Scrape Additional Market Data:** Expand dataset size beyond 272 records to improve model generalization.
2. **SHAP Interpretability:** Integrate SHAP waterfall plots into the Streamlit UI for individual prediction explanations.
3. **Automated Retraining:** Setup automated CI/CD retraining pipelines using GitHub Actions.
