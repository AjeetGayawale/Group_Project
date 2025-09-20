# app.py
import streamlit as st
import pickle
import pandas as pd
from fpdf import FPDF
from datetime import datetime

# ---- Load pipeline (preprocessing + model) ----
with open("churnmodel_fix.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction Dashboard")
st.write("Fill the form and click Predict. A downloadable PDF report will be generated.")

# ---- UI: collect all standard Telco features (about 20 fields) ----
st.markdown("### Customer details")

col1, col2 = st.columns(2)
with col1:
    customer_id = st.text_input("Customer ID (optional)", value="")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])  # will map to 0/1
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

col3, col4 = st.columns(2)
with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

st.markdown("### Billing")
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, format="%.2f")
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0, format="%.2f")

# ---- Build DataFrame matching original dataset column names ----
def build_input_df():
    data = {
        # keep same column names (case-sensitive) as in training
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    # Remove customerID column if your pipeline wasn't trained with it (safe to keep, column transformer will ignore unknown columns)
    return pd.DataFrame([data])

# ---- PDF report generator ----
def generate_pdf(customer_dict, pred_label, pred_proba=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Telco Customer Churn Prediction Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Customer Input:", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in customer_dict.items():
        # keep lines short
        pdf.multi_cell(0, 6, txt=f"{k}: {v}")

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    result_text = "LIKELY TO CHURN ❌" if pred_label == 1 else "NOT LIKELY TO CHURN ✅"
    pdf.set_text_color(200, 30, 30) if pred_label == 1 else pdf.set_text_color(30, 130, 30)
    pdf.cell(0, 8, f"Prediction: {result_text}", ln=True)

    if pred_proba is not None:
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 7, f"Probability of churn: {pred_proba:.2%}", ln=True)

    filename = f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ---- Predict & generate report ----
if st.button("🔍 Predict"):
    X_new = build_input_df()

    # Show input summary for user
    st.subheader("Input Summary")
    # show all fields except empty customerID
    display_df = X_new.T.rename(columns={0: "Value"})
    if display_df.index.str.lower().tolist().count("customerid") and not customer_id:
        display_df = display_df.drop(index="customerID", errors="ignore")
    st.table(display_df)

    try:
        # Make prediction with pipeline
        pred = pipeline.predict(X_new)[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_new)[0][1]
            except Exception:
                proba = None

        # Show result
        if pred == 1:
            st.error("❌ This customer is likely to churn.")
        else:
            st.success("✅ This customer is not likely to churn.")

        
    finally:
        st.write("Prediction attempt finished.")

    