# 🚗 Used Car Price Prediction

This project predicts the **price of used cars** using machine learning models.  
It estimates prices based on attributes like **car make, age, mileage, fuel type, engine specifications, and more**, and demonstrates **before and after hyperparameter tuning** results through a **Streamlit web application**.

---

## 🔧 Tech Stack

- **Programming Language:** Python
- **Libraries & Tools:** Pandas, NumPy, Scikit-learn, Joblib, Streamlit

---

## 📌 Features

- Loaded dataset and extracted **features relevant to car pricing**
- Trained and evaluated:
  - **Linear Regression (Ridge)** (before & after tuning)
  - **Random Forest Regression** (before & after tuning)
  - **XGBoost Regression** (before & after tuning)
- Compared **performance metrics**: R², MAE, RMSE, and ±10% tolerance
- Developed a **Streamlit UI (`app.py`)** for real-time car price prediction
- Compact input form for user to enter **car details** and get predicted prices from all models

---

## 🚀 How to Run

### 1️⃣ Clone the repository and navigate into it

```
git clone https://github.com/PranavSharma195/Artificial_Intelligence.git
cd Artificial_Intelligence
```

### 2️⃣ Install required packages

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```
streamlit run app.py
```

## 📁 Project Structure

Artificial_Intelligence/  
├── app.py - Streamlit application for real-time car price prediction  
├── 23048577_Pranav_Sharma.ipynb - Model training, evaluation, and analysis  
├── car_dataset.csv - Original dataset containing car features and prices  
├── lr_model_before.pkl - Linear Regression model before hyperparameter tuning  
├── lr_model_after.pkl - Linear Regression model after hyperparameter tuning  
├── rf_model_before.pkl - Random Forest model before hyperparameter tuning  
├── rf_model_after.pkl - Random Forest model after hyperparameter tuning  
├── xgb_model_before.pkl - XGBoost model before hyperparameter tuning  
├── xgb_model_after.pkl - XGBoost model after hyperparameter tuning  
├── feature_columns.pkl - List of features used for prediction  
├── X_test.pkl - Test set features for model evaluation  
├── y_test.pkl - Test set labels (actual car prices)  
├──README.md - This documentation file

---

## 📈 Future Improvements

- Include **regional pricing trends** and supply-demand factors to improve accuracy
- Adapt the system to **country-specific pricing rules** (e.g., pending services, insurance status in Nepal)
- Deploy as a **real-time web application** for continuous car price prediction
- Explore **advanced ML models or ensemble techniques** for better prediction performance
- Integrate with **used car marketplaces** to provide automated price suggestions
- Add **visualizations** such as price distributions, feature importance, and model comparison charts
- Include **user authentication** and save past predictions for repeat users
- Incorporate **live dataset updates** from online car listings for more accurate predictions
- Implement **mobile-friendly UI** for better accessibility

---

## 🤝 Contributions

- Feel free to fork the repository and open a pull request to improve the models or app.

---

## 📬 Contact

Created by **Pranav Sharma** – feel free to reach out!
