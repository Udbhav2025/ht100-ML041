# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "model.pkl"

app = Flask(__name__, template_folder="templates", static_folder="static")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run train_model.py to create it (or move it here).")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact['pipeline']
FEATURES = artifact['features']  # ordered feature names expected by the model

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/predict', methods=['POST'])
def predict():
    # Accept JSON or form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    # Build input row in FEATURES order; missing features => np.nan
    row = {}
    for f in FEATURES:
        found = None
        for k in data.keys():
            if k.lower() == f.lower():
                found = k
                break
        val = data.get(found if found is not None else f, None)
        try:
            val = float(val) if (val is not None and str(val) != "") else np.nan
        except:
            val = np.nan
        row[f] = val

    X = pd.DataFrame([row], columns=FEATURES)

    try:
        proba = pipeline.predict_proba(X)[0].tolist()
        pred = int(pipeline.predict(X)[0])
        return jsonify({"prediction": pred, "probabilities": proba, "features_used": FEATURES})
    except AttributeError:
        pred = pipeline.predict(X)[0].item()
        return jsonify({"prediction": pred, "features_used": FEATURES})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
