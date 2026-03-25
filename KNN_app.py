import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------- Load Data ----------
df = pd.read_csv("task1_dataset.csv")

# ---------- Handle Missing Values ----------
df['income'].fillna(df['income'].median(), inplace=True)
df['loan_amount'].fillna(df['loan_amount'].median(), inplace=True)
df['credit_score'].fillna(df['credit_score'].median(), inplace=True)
df['annual_spend'].fillna(df['annual_spend'].median(), inplace=True)

# ---------- Encode Categorical ----------
cat = ['city', 'employment_type', 'loan_type']
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[cat])
encoded_cols = encoder.get_feature_names_out(cat)

encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

df = df.drop(cat, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# ---------- Date Processing ----------
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day
df['year'] = pd.to_datetime(df['date']).dt.year
df = df.drop('date', axis=1)

# ---------- Scaling ----------
scale_cols = ['income','loan_amount','credit_score',
              'num_transactions','annual_spend']

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ---------- Split ----------
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Model ----------
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ======================================================
#                STREAMLIT UI
# ======================================================

st.title("💰 Customer Target Prediction App")

st.write("Enter customer details to predict target value")

# ---------- User Inputs ----------
income = st.number_input("Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0.0)
num_transactions = st.number_input("Number of Transactions", min_value=0)
annual_spend = st.number_input("Annual Spend", min_value=0.0)

city = st.selectbox("City", encoder.categories_[0])
employment_type = st.selectbox("Employment Type", encoder.categories_[1])
loan_type = st.selectbox("Loan Type", encoder.categories_[2])

date = st.date_input("Date")

# ---------- Predict ----------
if st.button("Predict"):

    # Create input dataframe
    input_df = pd.DataFrame({
        'income':[income],
        'loan_amount':[loan_amount],
        'credit_score':[credit_score],
        'num_transactions':[num_transactions],
        'annual_spend':[annual_spend]
    })

    # Date features
    input_df['month'] = date.month
    input_df['day'] = date.day
    input_df['year'] = date.year

    # Encode categorical
    cat_input = pd.DataFrame({
        'city':[city],
        'employment_type':[employment_type],
        'loan_type':[loan_type]
    })

    encoded_input = encoder.transform(cat_input)
    encoded_input_df = pd.DataFrame(
        encoded_input,
        columns=encoded_cols
    )

    input_df = pd.concat([input_df, encoded_input_df], axis=1)

    # Match training columns
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # Scale numeric columns
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Target Value: {prediction:.2f}")