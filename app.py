from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime
import time

app = Flask(__name__)

MODEL_PATH = "fraud_detection_minimal.pkl"
EXCEL_PATH = "Incident.xlsx"

# === Load model and data safely ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ Model file not found. Please run training first.")

if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError("⚠️ Dummy insurance data file missing.")

model = joblib.load(MODEL_PATH)
dummy_data = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# Convert policy date to datetime
dummy_data["policy_bind_date"] = pd.to_datetime(dummy_data["policy_bind_date"], errors="coerce")
min_date = dummy_data["policy_bind_date"].min()

@app.route("/")
def home():
    return jsonify({"message": "✅ Insurance Trust Score API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        policy_number = data.get("policy_number")
        incident_date = data.get("incident_date")
        property_claim = data.get("property_claim")

        if not (policy_number and incident_date and property_claim):
            return jsonify({"error": "Missing required fields."}), 400

        # Simulate a small wait to handle async requests smoothly
        time.sleep(2)

        # Find matching policy
        policy_row = dummy_data[dummy_data["policy_number"] == policy_number]
        if policy_row.empty:
            return jsonify({"error": "Policy number not found in dummy data."}), 404

        # Convert incident_date to numeric difference
        incident_dt = datetime.strptime(incident_date, "%Y-%m-%d")
        policy_bind_date = policy_row.iloc[0]["policy_bind_date"]
        policy_bind_days = (policy_bind_date - min_date).days
        incident_days = (incident_dt - min_date).days

        model_input = pd.DataFrame([{
            "policy_number": policy_number,
            "policy_bind_date": policy_bind_days,
            "incident_date": incident_days,
            "property_claim": property_claim,
            "total_claim_amount": property_claim
        }])

        fraud_prob = model.predict_proba(model_input)[0][1] * 100
        trust_score = 100 - fraud_prob

        return jsonify({
            "policy_number": policy_number,
            "trust_score": round(trust_score, 2),
            "incident_date": incident_date
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Detect if running under Gunicorn or standalone Flask
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
