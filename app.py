import streamlit as st
import pandas as pd
import joblib
import plotly.express as px



# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model = joblib.load("churn_model.pkl")

try:
    df = pd.read_csv("customer_churn_data.csv")
except FileNotFoundError:
    df = None


# =========================================================
# HEADER
# =========================================================

st.title("📊 Customer Churn Prediction")
st.write(
    "Analyze customer information and predict whether a customer "
    "is likely to churn or stay."
)

# =========================================================
# DASHBOARD STATISTICS
# =========================================================

if df is not None:

    total_customers = len(df)
    churned_customers = (df["Churn"] == "Yes").sum()
    stayed_customers = (df["Churn"] == "No").sum()
    churn_rate = (churned_customers / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "⚠️ Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "✅ Stayed Customers",
            f"{stayed_customers:,}"
        )

    with col4:
        st.metric(
            "📈 Churn Rate",
            f"{churn_rate:.1f}%"
        )


# =========================================================
# CHARTS
# =========================================================

if df is not None:

    st.subheader("📈 Customer Churn Overview")

    chart_col1, chart_col2 = st.columns(2)

    # ---------------- PIE CHART ----------------

    with chart_col1:

        churn_counts = df["Churn"].value_counts().reset_index()

        churn_counts.columns = ["Churn", "Customers"]

        fig_pie = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            title="Customer Churn Distribution",
            hole=0.35
        )

        fig_pie.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )


    # ---------------- BAR CHART ----------------

    with chart_col2:

        contract_churn = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig_bar = px.bar(
            contract_churn,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Churn by Contract Type"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("🧾 Customer Information")
st.write("Enter the details of the customer below.")
# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=1
    )


with col2:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step = 1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=50.0,
        step = 1.0
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )


with col3:

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )


# =========================================================
# ROW 2
# =========================================================

col4, col5, col6 = st.columns(3)


with col4:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )


with col5:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )


with col6:

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


# =========================================================
# ROW 3
# =========================================================

col7, col8, col9 = st.columns(3)


with col7:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer",
            "Credit card",
            "Electronic check",
            "Mailed check"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

col1, col2, col3 = st.columns([2, 1, 2])

with col2:

       predict_button = st.button("🔍 Predict")

if predict_button:

    # Create customer dataframe
    customer = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method]
    })

    # Encoding
    customer = pd.get_dummies(
        customer,
        drop_first=True
    )

    # Model columns
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

    # Match model columns
    customer = customer.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(customer)[0]

    # Probability
    probabilities = model.predict_proba(customer)[0]

    stay_probability = round(probabilities[0] * 100)
    churn_probability = round(probabilities[1] * 100)

    st.divider()
    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "✅ Stay Probability",
            f"{stay_probability:.1f}%"
        )

    with col2:
        st.metric(
            "⚠️ Churn Probability",
            f"{churn_probability:.1f}%"
        )

    if prediction == 1:
        st.error("⚠️ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")


#================================================================================================================





        st.markdown(
    '<div class="section-title">Customer Churn Distribution</div>',
    unsafe_allow_html=True
)

st.image("images/diffrence.png",width=800)


st.markdown(
    '<div class="section-title">Top Importance Features</div>',
    unsafe_allow_html=True
)

st.image("images/topfeatures.png",use_container_width=True)

