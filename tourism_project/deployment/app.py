import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="wash9968/customer-purchase-prediction-model", filename="best_predict_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Product Purchase Prediction App")
st.write("The Product Purchase Prediction App is an internal tool for tourism company staff that predicts whether customers will purchase product based on their details.")
st.write("Kindly enter the customer details to check whether they will purchase product.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.number_input("CityTier (Tier 1 > Tier 2 > Tier 3)", min_value=1, max_value=3, value = 1)
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=1, max_value=3, value = 1)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=3, max_value=5, value = 3)
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=1, value=2)
Passport = st.selectbox("Whether the customer holds a valid passport ", ["No","Yes"], index=1)	
OwnCar = st.selectbox("Whether the customer owns a car ", ["No","Yes"], index=1)
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer",min_value=0, value=0)
MonthlyIncome = st.number_input("Gross monthly income of the customer", value=17000)
PitchSatisfactionScore = st.number_input("Score indicating the customer's satisfaction with the sales pitch", min_value=1, max_value=5, value = 3)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch", min_value=1, value=3)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer", min_value=1, value=12)
TypeofContact = st.selectbox("The method by which the customer was contacted (Company Invited or Self Inquiry)", ["Company Invited", "Self Inquiry"], index=1)
Occupation = st.selectbox("Customer's occupation (e.g., Salaried, Freelancer)", ["Salaried", "FreeLancer", "Small Business", "Large Business"], index=0)
Gender = st.selectbox("Gender of the customer (Male, Female)", ["Male", "Female"], index=0)
MaritalStatus = st.selectbox("Marital status of the customer (Single, Married, Divorced)", ["Single", "Married", "Divorced"], index=0)
Designation = st.selectbox("Customer's designation in their current organization", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)
ProductPitched = st.selectbox("The type of product pitched to the customer", ["Basic", "Standard", "Deluxe","Super Deluxe",  "King"], index=0)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
  'Age': Age,
  'CityTier': CityTier,
  'NumberOfPersonVisiting': NumberOfPersonVisiting,
  'PreferredPropertyStar': PreferredPropertyStar,
  'NumberOfTrips': NumberOfTrips,
  'Passport': 1 if Passport == "Yes" else 0,
  'OwnCar': 1 if OwnCar == "Yes" else 0,
  'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
  'MonthlyIncome': MonthlyIncome,
  'PitchSatisfactionScore': PitchSatisfactionScore,
  'NumberOfFollowups': NumberOfFollowups,
  'DurationOfPitch': DurationOfPitch,
  'TypeofContact': TypeofContact,
  'Occupation': Occupation,
  'Gender': Gender,
  'MaritalStatus': MaritalStatus,
  'Designation': Designation,
  'ProductPitched': ProductPitched
}])

# Predict button
if st.button("Predict"):
    prediction = (model.predict_proba(input_data)[0, 1]).astype(int)
    result = "take product" if prediction == 1 else "not take product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
