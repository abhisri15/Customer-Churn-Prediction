import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
best_model = joblib.load('RandomSearchModel.pkl')

# Create a function to predict churn
def predict_churn(features):
    prediction = best_model.predict(features)
    return prediction

# Create the Streamlit app
def main():
    st.set_page_config(page_title='Customer Churn Prediction', layout='centered')
    st.title('Customer Churn Prediction')

    st.write('Enter the following details:')
    age = st.number_input('Age', min_value=0)
    subscription_months = st.number_input('Subscription Length (Months)', min_value=0)
    monthly_bill = st.number_input('Monthly Bill', min_value=0.0)
    total_usage_gb = st.number_input('Total Usage (GB)', min_value=0)
    gender_male = st.selectbox('Gender', ['Male', 'Female'])
    location = st.selectbox('Location', ['Houston', 'Los Angeles', 'Miami', 'New York'])

    # Calculate Feature Ratios and Differences
    usage_per_month = total_usage_gb / subscription_months
    bill_usage_difference = monthly_bill - total_usage_gb

    # Encode categorical variables
    gender_encoded = 1 if gender_male == 'Male' else 0
    location_encoded = 1 if location != 'Houston' else 0
    if location == 'Los Angeles':
        location_encoded = 2
    elif location == 'Miami':
        location_encoded = 3

    # Categorize age
    if 0 <= age <= 30:
        age_category_encoded = 0
    elif 30 < age <= 60:
        age_category_encoded = 1
    else:
        age_category_encoded = 2

    # Create a feature DataFrame
    features = pd.DataFrame({
        'Age': [age],
        'Subscription_Length_Months': [subscription_months],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [total_usage_gb],
        'Interaction_Subscription_Bill': [subscription_months * monthly_bill],
        'Log_Total_Usage_GB': [np.log(total_usage_gb + 1)],
        'Gender_Male': [gender_encoded],
        'Location_Houston': [location_encoded == 0],
        'Location_Los Angeles': [location_encoded == 2],
        'Location_Miami': [location_encoded == 3],
        'Location_New York': [location_encoded == 1],
        'Age_Category_Adult': [age_category_encoded == 1],
        'Age_Category_Senior': [age_category_encoded == 2],
        'Usage_Per_Month': [usage_per_month],
        'Bill_Usage_Difference': [bill_usage_difference]
    })

    if st.button('Predict Churn'):
        prediction = predict_churn(features)
        if prediction[0] == 1:
            st.error('Churn: Customer is likely to churn')
        else:
            st.success('No Churn: Customer is likely to stay')

if __name__ == '__main__':
    main()
