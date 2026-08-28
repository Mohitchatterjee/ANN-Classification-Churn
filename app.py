import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle



# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')


# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)


# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 99)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

# Preprocess user input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

# If the encoder was a LabelEncoder fitted on the training 'Gender' values:
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

geo_arr = onehot_encoder_geography.transform([[geography]]).toarray()
geo_cols = onehot_encoder_geography.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_arr, columns=geo_cols, index=input_data.index)

# Combine and drop the original Geography column
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale the input data using the loaded scaler
input_data = scaler.transform(input_data)


# Predict the churn probability using the trained model
churn_probability = model.predict(input_data) 

prediction_percentage = churn_probability[0][0] * 100
if prediction_percentage >= 50:
    st.write(f"The customer is likely to churn with a probability of {prediction_percentage:.2f}%.")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_percentage:.2f}%.")