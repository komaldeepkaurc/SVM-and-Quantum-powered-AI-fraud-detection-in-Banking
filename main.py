import numpy as np
import streamlit as st
from sklearn.svm import SVC, OneClassSVM
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header

# Define coordinates for example cities (simplified, not actual GPS)
geo_coordinates = {
    "Delhi": (28.61, 77.20),
    "Mumbai": (19.07, 72.87),
    "London": (51.50, -0.12),
    "New York": (40.71, -74.00),
    "Toronto": (43.65, -79.38),
    "Canada": (56.13, -106.35),
    "Sydney": (-33.87, 151.21),
    "Tokyo": (35.68, 139.76)
}

# Load or initialize blockchain-style fraud log
fraud_log_path = "fraud_log.json"
if not os.path.exists(fraud_log_path):
    with open(fraud_log_path, "w") as f:
        json.dump([], f)

def log_to_blockchain(entry):
    with open(fraud_log_path, "r+") as file:
        data = json.load(file)
        data.append(entry)
        file.seek(0)
        json.dump(data, file, indent=4)

# Dummy training data
X_train = np.array([
    [100, 10, 1, 0, 0.95],
    [250, 12, 2, 0, 0.90],
    [8000, 3, 12, 1, 0.2],
    [7000, 2, 15, 1, 0.1],
    [120, 14, 1, 0, 0.92],
    [6500, 1, 20, 1, 0.3],
])
y_train = [0, 0, 1, 1, 0, 1]  # 0 = legit, 1 = fraud

# Train Multi-Class SVM
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)

# Train One-Class SVM for anomaly detection (only on legit data)
ocsvm = OneClassSVM(gamma='auto').fit(X_train[np.array(y_train) == 0])

# Streamlit UI
st.set_page_config(page_title="Fraud Detection AI Demo", layout="centered", page_icon="🔐")
colored_header("Real-Time Fraud Detection (Patent Simulation)", description="Enhanced with One-Class SVM, Blockchain Logging & Intelligent Risk Scoring", color_name="violet-70")
st.caption("🔬 Developed by Komaldeep Kaur")
st.markdown("---")

st.subheader("📥 Enter Transaction Details")
amount = st.number_input("💰 Transaction Amount (₹)", value=100.0)
hour = st.slider("🕒 Hour (0-23)", 0, 23, 14)
frequency = st.slider("🔁 Frequency in Last Hour", 0, 20, 1)
location = st.selectbox("📍 Current Transaction Location", list(geo_coordinates.keys()))
biometric = st.slider("🧬 Biometrics Score (0-1)", 0.0, 1.0, 0.95)

loc_flag = 1 if location not in ["Delhi", "Mumbai"] else 0  # adjust based on usual location
input_data = np.array([[amount, hour, frequency, loc_flag, biometric]])

if st.button("🔎 Analyze Transaction"):
    risks = {}
    reasons = []

    # Risk logic
    risks["Amount"] = 0.8 if amount > 10000 else 0.5 if amount > 5000 else 0.2
    if risks["Amount"] > 0.7:
        reasons.append("💰 Unusually large transaction")

    risks["Hour"] = 0.7 if 0 <= hour <= 5 or hour >= 23 else 0.3
    if risks["Hour"] > 0.6:
        reasons.append("⏰ Transaction at odd hours")

    risks["Frequency"] = 0.9 if frequency >= 10 else 0.6 if frequency >= 5 else 0.2
    if risks["Frequency"] > 0.8:
        reasons.append("🔁 Too many transactions in short time")

    risks["Biometrics"] = 0.9 if biometric < 0.3 else 0.6 if biometric < 0.6 else 0.2
    if risks["Biometrics"] > 0.8:
        reasons.append("🧬 Behavioral mismatch")

    # Location jump logic
    location_jump_risk = 0
    if os.path.exists(fraud_log_path):
        with open(fraud_log_path, "r") as f:
            history = json.load(f)
            if history:
                last = history[-1]
                last_loc = last["location"]
                last_hour = last["hour"]
                hour_diff = abs(hour - last_hour)

                last_coords = geo_coordinates.get(last_loc)
                current_coords = geo_coordinates.get(location)
                if last_coords and current_coords and hour_diff <= 2:
                    dist = np.linalg.norm(np.array(last_coords) - np.array(current_coords))
                    if dist > 20:
                        location_jump_risk = 0.95
                        reasons.append("📍 Location jump detected in short time")
                    elif dist > 10:
                        location_jump_risk = 0.6

    risks["LocationJump"] = location_jump_risk

    # Overall risk score
    total_risk_score = sum(risks.values()) / len(risks)

    # Predictions
    class_pred = svc.predict(input_data)[0]
    class_prob = svc.predict_proba(input_data)[0][1] * 100
    ocsvm_pred = ocsvm.predict(input_data)[0]

    st.subheader("🔍 Results")
    if class_pred == 1 or ocsvm_pred == -1 or location_jump_risk > 0.8:
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        st.metric("Fraud Probability", f"{max(class_prob, location_jump_risk * 100):.2f}%")
    else:
        st.success("✅ Legitimate Transaction")
        st.metric("Fraud Probability", f"{class_prob:.2f}%")

    style_metric_cards(background_color="#f0f0f5", border_left_color="#f63366")

    if reasons:
        st.warning("**Reasons Detected:**")
        for r in reasons:
            st.markdown(f"- {r}")

    # Risk factor chart
    fig, ax = plt.subplots()
    ax.bar(risks.keys(), risks.values(), color=['#e63946' if r > 0.6 else '#2a9d8f' for r in risks.values()])
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color='orange', linestyle='--', label="High Risk Threshold")
    ax.set_ylabel("Risk Score (0-1)")
    ax.set_title("📊 Risk Factor Analysis")
    ax.legend()
    st.pyplot(fig)

    # Blockchain-style log entry
    log_entry = {
        "amount": amount,
        "hour": hour,
        "frequency": frequency,
        "location": location,
        "location_mismatch": loc_flag,
        "biometric_score": biometric,
        "predicted_fraud": bool(class_pred == 1 or ocsvm_pred == -1 or location_jump_risk > 0.8),
        "fraud_score": round(max(class_prob, location_jump_risk * 100), 2),
        "reasons": reasons,
        "timestamp": datetime.now().isoformat()
    }
    log_to_blockchain(log_entry)
    st.info("📦 Transaction securely logged to fraud ledger (simulated blockchain)")
