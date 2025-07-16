import streamlit as st
import numpy as np
import pandas as pd
import joblib

#load out saved components
model=joblib.load("gradient_boosting_model.pkl")
label_encoder=joblib.load('label_encoder.pkl')
scaler=joblib.load('scaler.pkl')


st.title('Glass Type Prediction App')
st.write('This model predicts the glass type according to the input features below. Please enter your values: ')

# the features you outline should match the features used during the training of the model

#input features

RI = st.slider('Refractive Index: ',1.5,1.8)
Na = st.slider('Sodium: ',10.0,15.0)
Mg = st.slider('Magnesium: ',0.0,4.0)
Al = st.slider('Aluminium: ',0.0,4.0)
Si = st.slider('Silicon: ',70.00,80.00)
K = st.slider('Potassium: ',10.00,18.00)
Ca = st.slider('Calcium: ',5.0,10.0)
Ba = st.slider('Barium: ',0.0,5.0)
Fe = st.slider('Iron: ',0.0,5.0)
#preparing input features for model
features=np.array([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
scaled_features=scaler.transform(features)

#prediction
if st.button('Predict Glass Type'):
    prediction_encoded=model.predict(scaled_features)
    prediction_label=label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f'Predicted glass type: {prediction_label}')