import streamlit as st 
import pandas as pd 
import joblib 

# Load trained model 
model = joblib.load("churn_model.pkl")

#Page configuration 
st.set_page_config(
    page_title = "Customer Churn Prediction" ,
    page_icon = "📊",
    layout = "centered"
)

#Title 

st.title(" 📊 Customer Churn Prediction")
st.write("Enter the customer details below to predict whether the customer will churn.")

st.divider()

#Customer details 
st.subheader("Customer Information")

gender = st.selectbox("Gender", ["Female", "Male"])

senior_citizen = st.selectbox(
  "Senior Citizen",
    [0,1]
    )

tenure = st.number_input(
    "Tenure (month)",
    min_value=0,
    max_value=100,
    value=1
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value = 0.0,
    value=50.0
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

phone_service = st.selectbox(
    "Phone Service",
["No", "Yes"]
)
multiple_lines = st.selectbox(
  "Multiply Lines",
["No phone services", "No", "Yes"]
 )

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes","No internet service"]
)
online_backup = st.selectbox(
    "Online Backup",
    ["No","Yes","No internet service"]
)
device_protection = st.selectbox(
    "Device Protection", 
    ["No", "Yes","No internet service"]
)
tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes","No internet Service"]
)
streaming_tv = st.selectbox(
    "Streaming Tv",
    ["No","Yes","No internet service"]
)
streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)
contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method  = st.selectbox(
    "Payment Method",
    [
        "Bank transfer",
        "Credit Card",
        "Electronic check",
        "Mailed check"
    ]
)

st.divider()

#Prediction

if st.button("🔍 Predict Churn"):

    #Created dataframe using the original 19 columns 
    customer = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges" :[monthly_charges],
        "TotalCharges" : [total_charges],
        "gender":[gender],
        "Partner":[partner],
        "Dependents": [dependents],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService" :[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup" : [online_backup],
        "DeviceProtection" :[device_protection],
        "TechSupport" :[tech_support],
        "StreamingTV" :[streaming_tv],
        "StreamingMovies" :[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling" :[paperless_billing],
        "PaymentMethod" : [payment_method]
    })


    #Apply the Same encoding used during the training 
    customer = pd.get_dummies(customer, drop_first=True)

    #Make sure the columns are exactly the same as the trained model 
    model_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

#Add missing columns and keep correct order 

    customer = customer.reindex(
    columns=model_columns,
    fill_value=0
     )

    #Prediction
    prediction = model.predict(customer)

    #Display result 
    if prediction[0] == 1:
     st.error(" ⚠️ Customer Will Churn")
    else:
     st.success(" ✅ Customer Will Stay")



