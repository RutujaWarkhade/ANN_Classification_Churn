import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the model
model = tf.keras.models.load_model('model.h5')

#load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_country.pkl', 'rb') as file:
    onehot_encoder_country = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')

# user input 
country = st.selectbox('country', onehot_encoder_country.categories_[0])
gender = st.selectbox('gender', label_encoder_gender.classes_)
age = st.slider('age', 18, 92)
balance = st.number_input('balance')
credit_score = st.number_input('credit_score')
estimated_salary = st.number_input('estimated_salary')
tenure = st.slider('tenure', 0, 10)
products_number = st.slider('number of products', 1, 4)
credit_card = st.selectbox('Has credit card', [0, 1])
active_member = st.selectbox('Is Active member',[0, 1])

# prepare input data
input_data = pd.DataFrame({
    'credit_score':[credit_score],
    'gender':[label_encoder_gender.transform([gender])[0]],
    'age':[age],
    'tenure':[tenure],
    'balance':[balance],
    'products_number':[products_number],
    'credit_card':[credit_card],
    'active_member':[active_member],
    'estimated_salary':[estimated_salary]

})

#one-hot encode 'country'
country_encoded = onehot_encoder_country.transform([[country]]).toarray()
country_encoded_df = pd.DataFrame(country_encoded, columns=onehot_encoder_country.get_feature_names_out(['country']))

## combine one hot encoder columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True), country_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

