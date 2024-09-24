import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Analysis')

# Collect user input
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
Usage_Frequency = st.number_input('Usage Frequency')	
Support_Calls = st.number_input('Support Calls')	
Payment_Delay = st.number_input('Payment Delay')	
Last_Interaction = st.number_input('Last Interaction')
Total_Spend = st.number_input('Total Spend')	
contract_length = st.selectbox('Contract Length', onehot_encoder.categories_[0])
tenure = st.slider('Tenure', 0, 100)

# Encode categorical variables
gender_encoded = label_encoder.transform([gender])[0]  # Label encode Gender
contract_length_encoded = onehot_encoder.transform([[contract_length]]).toarray()  # One-hot encode Contract Length

# Create the input DataFrame with the correct column names (matching those used during model training)
input_df = pd.DataFrame([{
    'Gender': gender_encoded,
    'Age': age,
    'Usage Frequency': Usage_Frequency,   # Match feature name used during model training
    'Support Calls': Support_Calls,       # Match feature name used during model training
    'Payment Delay': Payment_Delay,       # Match feature name used during model training
    'Last Interaction': Last_Interaction, # Match feature name used during model training
    'Total Spend': Total_Spend,           # Match feature name used during model training
    'Tenure': tenure
}])

# Convert contract length to DataFrame and concatenate with the input data
contract_length_df = pd.DataFrame(contract_length_encoded, columns=onehot_encoder.get_feature_names_out(['Contract Length']))
data_updated = pd.concat([input_df, contract_length_df], axis=1)

# Ensure the column order matches the scaler's expected feature names
expected_feature_names = scaler.feature_names_in_  # Assuming the scaler was fit with feature names

# Reorder the columns in `data_updated` to match the order of features used in the scaler
data_updated = data_updated[expected_feature_names]

# Scale the input data
input_scaled = scaler.transform(data_updated)

# Predict the churn probability
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

# Display the prediction result
if prediction_prob >= 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.write(f"The customer is not likely to churn with a probability of {1 - prediction_prob:.2f}")
