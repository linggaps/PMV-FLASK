from flask import Flask, request, jsonify, Response
import numpy as np
import joblib
import os
import csv
from io import StringIO
from datetime import datetime, timedelta  # <<< tambahkan timedelta
from flask_cors import CORS
from supabase import create_client, Client

# 🔧 Konfigurasi Supabase
SUPABASE_URL = "https://qyydmxvdbliivheskrgn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF5eWRteHZkYmxpaXZoZXNrcmduIiwicm9sZSI6ImFub24i" \
"LCJpYXQiOjE3NDcyMDI3OTIsImV4cCI6MjA2Mjc3ODc5Mn0.1cwUyDv9Dwoc6QMcwuduFaiFXj_Ub_9cBKin69Mw_Hw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔧 Model & Scaler
MODEL_PATH = os.path.join(os.getcwd(), "model_pmv.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")

application = Flask(__name__)
CORS(application)

# Nonaktifkan GPU TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Error saat memuat model atau scaler: {e}")
    model, scaler = None, None

# 🔄 Fungsi Thermal Comfort
def get_thermal_comfort_status(pmv_raw):
    # Batas Max and Min PMV
    pmv = max(min(pmv_raw, 3.0), -3.0)

    # Format 2 angka dibelakang koma 
    pmv_rounded = round(pmv, 2)

    # Status thermal comfort dengan logika baru
    if pmv_rounded >= 3.0:  
        status = "Hot"
    elif pmv_rounded >= 2.0:
        status = "Warm"
    elif pmv_rounded >= 1.0:
        status = "A bit Warm"
    elif pmv_rounded >= -1.0:
        status = "Normal"
    elif pmv_rounded > -2.0:
        status = "A Bit Cool"
    elif pmv_rounded > -3.0:
        status = "Cool"
    else:
        status = "Cold"

    return pmv_rounded, status

# 🚀 Endpoint: Root
@application.route('/')
def home():
    return jsonify({"message": "PMV Prediction API is running!"})

# 🚀 Endpoint: Predict
@application.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded"}), 500

    data = request.get_json()
    fields = ["temperature", "humidity", "air_flow", "mrt"]

    if not all(field in data for field in fields):
        return jsonify({"error": "Missing required fields!"}), 400

    try:
        input_values = [float(data[field]) for field in fields]
    except ValueError:
        return jsonify({"error": "Invalid input values!"}), 400

    input_scaled = scaler.transform([input_values])
    pmv_value_raw = float(model.predict(input_scaled)[0])
    pmv_value, tc_status = get_thermal_comfort_status(pmv_value_raw)

    # ⬇️ Simpan ke Supabase dengan waktu +7 jam (WIB)
    now = (datetime.utcnow() + timedelta(hours=7)).isoformat()  # <<< offset +7 jam

    payload = {
        "time": now,
        "temperature": input_values[0],
        "humidity": input_values[1],
        "air_flow": input_values[2],
        "mrt": input_values[3],
        "pmv": pmv_value,
        "thermal_comfort": tc_status
    }

    try:
        supabase.table("sensor_data").insert(payload).execute()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"pmv": pmv_value, "thermal_comfort": tc_status})

# 🚀 Endpoint: Get Latest Sensor Data
@application.route('/sensor-data', methods=['GET'])
def get_latest_data():
    result = supabase.table("sensor_data").select("*").order("time", desc=True).limit(1).execute()
    if result.data:
        return jsonify(result.data[0])
    return jsonify({"error": "No data found!"}), 404

# 🚀 Endpoint: Get All History
@application.route('/sensor-data/history', methods=['GET'])
def get_history():
    result = supabase.table("sensor_data").select("*").order("time", desc=True).execute()
    return jsonify(result.data)

# 🚀 Endpoint: Delete by ID
@application.route('/sensor-data/<int:id>', methods=['DELETE'])
def delete_data(id):
    result = supabase.table("sensor_data").delete().eq("id", id).execute()
    if result.count > 0:
        return jsonify({"message": "Data deleted!"})
    return jsonify({"error": "Data not found!"}), 404

# 🚀 Endpoint: Export to CSV
@application.route('/export-csv', methods=['GET'])
def export_csv():
    result = supabase.table("sensor_data").select("*").order("time", desc=True).execute()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Time', 'Temperature', 'Humidity', 'Air Flow', 'MRT', 'PMV', 'Thermal Comfort'])

    for row in result.data:
        writer.writerow([
            row['id'], row['time'], row['temperature'], row['humidity'],
            row['air_flow'], row['mrt'], row['pmv'], row['thermal_comfort']
        ])

    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sensor_data.csv"}
    )

# ▶️ Main
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    application.run(host='0.0.0.0', port=port)
