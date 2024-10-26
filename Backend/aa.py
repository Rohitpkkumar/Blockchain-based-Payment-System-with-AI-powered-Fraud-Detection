# fina code

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

# Load the pre-trained pipeline and model
preprocessor, model = joblib.load('random_forest_fraud_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Configuration for OnDemand API
ONDEMAND_API_KEY = '9QD3Lw1ywQOrEqragnlOWFi8rmhM1YGL'  # Replace with your OnDemand API Key
PLUGIN_ID = 'plugin-1718195424'  # Replace with the Solscan plugin ID from OnDemand
ONDEMAND_BASE_URL = 'https://api.on-demand.io/'

# Define the expected feature names for prediction
expected_features = ["sender_wallet", "recipient_wallet", "timestamp", "transaction_type", "amount", "gas_fee"]

# Fetch transactions using the Solscan plugin on OnDemand
def fetch_transactions_from_plugin(transaction_hash):
    url = "https://api.on-demand.io/plugin/v1/list?pluginIds=plugin-1718195424"
    headers = {
        "accept": "application/json",
        "apikey": ONDEMAND_API_KEY
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed with status code {response.status_code}", "details": response.text}

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    # Check for missing features
    missing_features = [feature for feature in expected_features if feature not in data]
    if missing_features:
        return jsonify({"error": f"Missing values for {', '.join(missing_features)}"}), 400
    
    # Prepare input data as a DataFrame with expected features
    try:
        input_data = pd.DataFrame([[data[feature] for feature in expected_features]], columns=expected_features)
    except ValueError:
        return jsonify({"error": "All feature values should be provided correctly."}), 400
    
    # Apply the preprocessor to the input data
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_preprocessed)[0]
    
    # Define fraud types based on model classes
    fraud_types = {
        0: "Not Fraud",
        1: "Type 1 Fraud",
        2: "Type 2 Fraud",
        3: "Type 3 Fraud",
        4: "Type 4 Fraud",
        5: "Type 5 Fraud"
    }
    
    # Map prediction to fraud type
    prediction_label = fraud_types.get(prediction, "Unknown")
    
    # Return result as JSON
    return jsonify({"prediction": prediction_label})

@app.route('/test_solscan', methods=['POST'])
def test_solscan():
    data = request.get_json()
    transaction_hash = data.get("transaction_hash")

    if not transaction_hash:
        return jsonify({"error": "Transaction hash is required."}), 400

    # Fetch transaction details from the Solscan plugin on OnDemand
    transaction_data = fetch_transactions_from_plugin(transaction_hash)
    
    if "error" in transaction_data:
        return jsonify(transaction_data), 500

    # Format the response if necessary
    return jsonify({"transaction_data": transaction_data})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
