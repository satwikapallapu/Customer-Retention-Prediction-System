from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- LOAD SAVED OBJECTS ----------------
with open("best_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("cat_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("feature_column.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("Churn_Prediction_Best_Model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "gender": request.form["gender"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "SIM": request.form["SIM"],
            "DeviceType": request.form["DeviceType"],
            "Region": request.form["Region"],
            "tenure": float(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
        }

        df = pd.DataFrame([data])

        # Encode categorical
        cat_cols = df.select_dtypes(include="object").columns
        df[cat_cols] = encoder.transform(df[cat_cols])

        # Align features
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        result = "Churn" if prediction == 1 else "Not Churn"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{prob*100:.2f}%"
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)