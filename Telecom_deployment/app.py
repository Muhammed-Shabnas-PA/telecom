from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Define input columns and mappings
binary_columns = {
    "SeniorCitizen": {"Yes": 1, "No": 0},
    "Partner": {"Yes": 1, "No": 0},
    "Dependents": {"Yes": 1, "No": 0},
    "OnlineSecurity": {"Yes": 1, "No": 0},
    "OnlineBackup": {"Yes": 1, "No": 0},
    "DeviceProtection": {"Yes": 1, "No": 0},
    "TechSupport": {"Yes": 1, "No": 0},
    "PaperlessBilling": {"Yes": 1, "No": 0},
}

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

categorical_columns = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

selected_columns = list(binary_columns.keys()) + numeric_columns + list(categorical_columns.keys())

# Initialize label encoders for categorical columns
contract_encoder = LabelEncoder()
payment_method_encoder = LabelEncoder()

# Fit label encoders on the categorical columns (make sure to fit once with the available data)
contract_encoder.fit(categorical_columns["Contract"])
payment_method_encoder.fit(categorical_columns["PaymentMethod"])

@app.route('/')
def home():
    return render_template(
        'index.html',
        binary_columns=binary_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data and process it
        input_data = {}

        # Process binary columns
        for column, mapping in binary_columns.items():
            value = request.form.get(column)
            if value is not None:
                input_data[column] = mapping.get(value, 0)  # Default to 0 if not found

        # Process numeric columns
        for column in numeric_columns:
            value = request.form.get(column)
            if value:
                input_data[column] = float(value)

        # Process categorical columns
        if "Contract" in request.form:
            input_data["Contract"] = contract_encoder.transform([request.form["Contract"]])[0]
        if "PaymentMethod" in request.form:
            input_data["PaymentMethod"] = payment_method_encoder.transform([request.form["PaymentMethod"]])[0]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Align input data with model features (ensure all required columns are present)
        model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else selected_columns
        for column in model_features:
            if column not in input_df.columns:
                input_df[column] = 0  # Add missing columns with default values

        input_df = input_df[model_features]  # Align columns with the model

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100  # Assuming binary classification

        # Return the result to the prediction page
        return render_template('predict.html', prediction=prediction, probability=probability)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
