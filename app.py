import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


MODEL_FILENAME = "churnmodel_fix.pkl"


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def prepare_input_dataframe(
    payload: Any,
    feature_names: Optional[List[str]],
    n_features_in: Optional[int],
) -> pd.DataFrame:
    # Accept either a single record or a batch under key "instances"
    instances: List[Any]
    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
    else:
        instances = [payload]

    if feature_names is not None:
        normalized: List[Dict[str, Any]] = []
        for item in instances:
            if not isinstance(item, dict):
                raise ValueError(
                    "Expected object(s) with named features because the model exposes feature_names_in_."
                )
            missing = [name for name in feature_names if name not in item]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            ordered = {name: item[name] for name in feature_names}
            normalized.append(ordered)
        return pd.DataFrame(normalized)

    # Fallback to positional features
    if n_features_in is None:
        rows: List[Dict[str, Any]] = []
        for item in instances:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, list):
                cols = [f"f{i}" for i in range(len(item))]
                rows.append({c: v for c, v in zip(cols, item)})
            else:
                raise ValueError("Each instance must be a dict or list")
        return pd.DataFrame(rows)

    rows_list: List[List[Any]] = []
    for item in instances:
        if not isinstance(item, list):
            raise ValueError(
                f"Expected list(s) of length {n_features_in} because the model does not expose feature names."
            )
        if len(item) != n_features_in:
            raise ValueError(
                f"Each list must have length {n_features_in}, got {len(item)}"
            )
        rows_list.append(item)
    cols = [f"f{i}" for i in range(n_features_in)]
    return pd.DataFrame(rows_list, columns=cols)


def main():
    st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
    st.title("📉 Customer Churn Prediction")
    st.caption("Model: " + MODEL_FILENAME)

    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

    with st.sidebar:
        st.header("Settings")
        with st.spinner("Loading model..."):
            model = load_model(model_path)
        feature_names_np = getattr(model, "feature_names_in_", None)
        feature_names = (
            list(map(str, feature_names_np)) if feature_names_np is not None else None
        )
        n_features_in = getattr(model, "n_features_in_", None)

        st.write("Input mode")
        input_mode = st.radio(
            "Choose input mode",
            options=["Manual", "JSON", "CSV"],
            horizontal=True,
        )

    def do_predict(X: pd.DataFrame) -> Dict[str, Any]:
        y_pred = model.predict(X)
        result: Dict[str, Any] = {
            "predictions": y_pred.tolist()
            if isinstance(y_pred, np.ndarray)
            else list(y_pred)
        }
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
                result["probabilities"] = y_proba.tolist()
            except Exception:
                pass
        return result

    if input_mode == "Manual":
        st.subheader("Manual Input")
        records: List[Dict[str, Any]] = []
        if feature_names is not None:
            with st.form("manual_form"):
                cols = st.columns(min(4, len(feature_names)))
                values: Dict[str, Any] = {}
                for idx, name in enumerate(feature_names):
                    with cols[idx % len(cols)]:
                        values[name] = st.text_input(name, value="")
                batch = st.number_input("Batch size", min_value=1, max_value=1000, value=1)
                submitted = st.form_submit_button("Predict")
            if submitted:
                # replicate values for batch
                records = [values for _ in range(batch)]
                try:
                    X = prepare_input_dataframe({"instances": records}, feature_names, n_features_in)
                    output = do_predict(X)
                    st.success("Prediction complete")
                    st.json(output)
                except Exception as e:
                    st.error(str(e))
        else:
            st.info(
                "Model does not expose feature names. Enter a comma-separated list of values; set the expected length using the field below."
            )
            n_expected = st.number_input(
                "Expected feature length (n_features_in_)",
                min_value=1,
                value=int(n_features_in) if n_features_in else 3,
            )
            values_str = st.text_input("Values (comma-separated)", value="")
            batch = st.number_input("Batch size", min_value=1, max_value=1000, value=1)
            if st.button("Predict"):
                try:
                    values = [v.strip() for v in values_str.split(",") if v.strip() != ""]
                    if len(values) != n_expected:
                        raise ValueError(f"Expected {n_expected} values, got {len(values)}")
                    row = [try_cast_number(x) for x in values]
                    X = prepare_input_dataframe({"instances": [row for _ in range(batch)]}, None, n_expected)
                    output = do_predict(X)
                    st.success("Prediction complete")
                    st.json(output)
                except Exception as e:
                    st.error(str(e))

    elif input_mode == "JSON":
        st.subheader("JSON Upload or Paste")
        st.write("Use either a single object, a list, or {instances: [...]}.")
        uploaded = st.file_uploader("Upload JSON file", type=["json"])
        text = st.text_area("Or paste JSON here", height=200, value="")
        if st.button("Predict from JSON"):
            try:
                if uploaded is not None:
                    payload = json.load(uploaded)
                else:
                    payload = json.loads(text) if text.strip() else {}
                X = prepare_input_dataframe(payload, feature_names, n_features_in)
                output = do_predict(X)
                st.success("Prediction complete")
                st.json(output)
            except Exception as e:
                st.error(str(e))

    else:  # CSV
        st.subheader("CSV Upload")
        st.write("CSV should contain feature columns. If names are unknown, order must match model input.")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
                st.write("Preview:")
                st.dataframe(df.head())
                if st.button("Predict from CSV"):
                    X = df
                    # If model expects specific feature order, reorder when available
                    if feature_names is not None:
                        missing = [c for c in feature_names if c not in df.columns]
                        if missing:
                            raise ValueError(f"Missing columns in CSV: {missing}")
                        X = df[feature_names]
                    output = do_predict(X)
                    st.success("Prediction complete")
                    st.json(output)
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.caption("Tip: run with 'streamlit run app.py'")


def try_cast_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    # Try int then float
    try:
        iv = int(str(value))
        return iv
    except Exception:
        pass
    try:
        fv = float(str(value))
        return fv
    except Exception:
        return value


if __name__ == "__main__":
    main()

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

    