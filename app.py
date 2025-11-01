from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import time
import logging

app = Flask(__name__)

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Load Model & Dummy Data ===
MODEL_PATH = "fraud_detection_minimal.pkl"
EXCEL_PATH = "Incident.xlsx"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ Model file not found. Please train and export it first.")

if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError("⚠️ Dummy insurance data file missing (Incident.xlsx).")

model = joblib.load(MODEL_PATH)
dummy_data = pd.read_excel(EXCEL_PATH, engine="openpyxl")

dummy_data["policy_bind_date"] = pd.to_datetime(dummy_data["policy_bind_date"], errors="coerce")
min_date = dummy_data["policy_bind_date"].min()

@app.route("/")
def home():
    return jsonify({"message": "✅ Insurance Fraud Detection API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        policy_number = data.get("policy_number")
        incident_date = data.get("incident_date")
        property_claim = data.get("property_claim")

        if not (policy_number and incident_date and property_claim):
            return jsonify({"error": "Missing required fields."}), 400

        start_time = time.time()
        logging.info(f"📩 Received request for Policy #{policy_number}. Waiting for model result...")

        # --- Lookup policy data ---
        policy_row = dummy_data[dummy_data["policy_number"] == policy_number]
        if policy_row.empty:
            return jsonify({"error": "Policy number not found in dummy data."}), 404

        policy_bind_date = policy_row.iloc[0]["policy_bind_date"]
        policy_bind_days = (policy_bind_date - min_date).days

        # --- Prepare model input ---
        total_claim_amount = property_claim

        model_input = pd.DataFrame([{
            "policy_number": policy_number,
            "policy_bind_date": policy_bind_days,
            "incident_date": incident_date,
            "property_claim": property_claim,
            "total_claim_amount": total_claim_amount
        }])

        logging.info("🧠 Running model.predict_proba() — please wait...")

        # --- Actual prediction (waits until done) ---
        fraud_prob = model.predict_proba(model_input)[0][1] * 100
        trust_score = 100 - fraud_prob

        processing_time = round(time.time() - start_time, 2)
        logging.info(f"✅ Completed. Policy #{policy_number} trust score: {trust_score:.2f}% (Took {processing_time}s)")

        return jsonify({
            "policy_number": policy_number,
            "trust_score": round(trust_score, 2),
            "incident_date": incident_date,
            "processing_time_seconds": processing_time
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
