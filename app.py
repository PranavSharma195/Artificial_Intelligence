# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------
# 1. Page Configuration
# ---------------------------
st.set_page_config(page_title="Used Car Price Prediction", layout="wide")

# ---------------------------
# Student Info (Top Right - WHITE TEXT)
# ---------------------------
# ---------------------------
# Student Info (Top Left - White Text, Clean & Visible)
# ---------------------------
# ---------------------------
# Student Info (Top of Page, White Text)
# ---------------------------
st.markdown(
    """
    <div style="
        color: white;
        font-family: Arial, sans-serif;
        font-size: 20px;
        font-weight: bold;
        line-height: 1.5;
        margin-bottom: 10px;
        word-wrap: break-word;
        max-width: 100%;
    ">
        Presented By: Pranav Sharma | LMU ID: 23048577
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title & Description
# ---------------------------
st.title("🚗 Used Car Price Prediction Before & After Hyperparameter Tuning")

st.markdown("""
This application predicts the price of a used car using three machine learning models:

- **Linear Regression (Ridge)**
- **Random Forest**
- **XGBoost**

General evaluation metrics are shown first, followed by a compact input form.
""")

# ---------------------------
# 2. Load Models & Data
# ---------------------------
lr_before = joblib.load("lr_model_before.pkl")
lr_after = joblib.load("lr_model_after.pkl")

rf_before = joblib.load("rf_model_before.pkl")
rf_after = joblib.load("rf_model_after.pkl")

xgb_before = joblib.load("xgb_model_before.pkl")
xgb_after = joblib.load("xgb_model_after.pkl")

features = joblib.load("feature_columns.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

makes = ["BMW", "Audi", "Toyota", "Honda", "Ford", "Mercedes", "Hyundai", "Kia"]

# ---------------------------
# 3. Helper Functions
# ---------------------------
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    tol_acc = ((abs(y_true - y_pred) / y_true) < 0.1).mean()
    return r2, mae, rmse, tol_acc

def display_metrics_card(title, y_true, y_pred, color):
    r2, mae, rmse, tol = evaluate_model(y_true, y_pred)
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:18px; border-radius:12px;
                    box-shadow:2px 2px 10px gray; margin-bottom:15px; color:black;">
        <h4>{title}</h4>
        <p>
        R² Score: {r2:.4f}<br>
        MAE: {mae:.2f}<br>
        RMSE: {rmse:.2f}<br>
        Tolerance (±10%): {tol:.2%}
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def predict_price(model, input_df):
    return model.predict(input_df)[0]

def display_prediction_card(title, price, color):
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:22px; border-radius:12px;
                    box-shadow:2px 2px 10px gray; text-align:center; color:black;">
        <h4>{title}</h4>
        <h2>₹ {price:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# 4. General Evaluation Metrics
# ---------------------------
st.header("📊 General Evaluation Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    display_metrics_card("Linear Regression (Before Tuning)", y_test, lr_before.predict(X_test), "#FFD580")
    display_metrics_card("Linear Regression (After Tuning)", y_test, lr_after.predict(X_test), "#90EE90")

with col2:
    display_metrics_card("Random Forest (Before Tuning)", y_test, rf_before.predict(X_test), "#FFD580")
    display_metrics_card("Random Forest (After Tuning)", y_test, rf_after.predict(X_test), "#90EE90")

with col3:
    display_metrics_card("XGBoost (Before Tuning)", y_test, xgb_before.predict(X_test), "#FFD580")
    display_metrics_card("XGBoost (After Tuning)", y_test, xgb_after.predict(X_test), "#90EE90")

# ---------------------------
# 5. Compact Input Form
# ---------------------------
st.header("📝 Enter Car Details")

input_df = pd.DataFrame(0, index=[0], columns=features)

make_selected = st.selectbox("Car Make", makes)
make_col = f"Make_{make_selected}"
if make_col in input_df.columns:
    input_df[make_col] = 1

cols = st.columns(3)

for i, feature in enumerate(features):
    if feature.startswith("Make_"):
        continue

    col = cols[i % 3]
    min_val = int(X_test[feature].min())
    max_val = int(X_test[feature].max())
    val = col.number_input(feature, min_value=min_val, max_value=max_val, value=min_val)
    input_df[feature] = val

# ---------------------------
# 6. Prediction Section
# ---------------------------
if st.button("🚀 Predict Car Price"):
    st.header("💰 Predicted Prices")

    col1, col2, col3 = st.columns(3)

    with col1:
        display_prediction_card("Linear Regression (Before)", predict_price(lr_before, input_df), "#FFD580")
        st.write("")
        display_prediction_card("Linear Regression (After)", predict_price(lr_after, input_df), "#90EE90")

    with col2:
        display_prediction_card("Random Forest (Before)", predict_price(rf_before, input_df), "#FFD580")
        st.write("")
        display_prediction_card("Random Forest (After)", predict_price(rf_after, input_df), "#90EE90")

    with col3:
        display_prediction_card("XGBoost (Before)", predict_price(xgb_before, input_df), "#FFD580")
        st.write("")
        display_prediction_card("XGBoost (After)", predict_price(xgb_after, input_df), "#90EE90")