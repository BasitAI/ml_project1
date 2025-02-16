import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Streamlit App
model = joblib.load('premium_prediction_model.pkl')

st.title("Premium Prediction App")

st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 30)
    bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    genetical_risk = st.sidebar.slider('Genetical Risk', 0, 100, 50)
    income_level = st.sidebar.selectbox('Income Level', ['Low', 'Medium', 'High'])
    smoking_status = st.sidebar.selectbox('Smoking Status', ['Non-Smoker', 'Smoker'])
    employment_status = st.sidebar.selectbox('Employment Status', ['Salaried', 'Self-Employed', 'Unemployed'])

    data = {
        'age': age,
        'bmi': bmi,
        'genetical_risk': genetical_risk,
        'income_level': income_level,
        'smoking_status': smoking_status,
        'employment_status': employment_status
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

input_df = pd.get_dummies(input_df, drop_first=True)

model_features = joblib.load('model_features.pkl')
input_df = input_df.reindex(columns=model_features, fill_value=0)

prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f"Estimated Annual Premium Amount: ${prediction[0]:,.2f}")
