from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__) 

MODEL_PATH = "best_model_XGBoost.pkl" 
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        data = request.get_json()
        
        df_input = pd.DataFrame(data)

        if 'income' in df_input.columns:
            df_input['income'] = np.log1p(df_input['income'])

        predictions = model.predict(df_input)
        probabilities = model.predict_proba(df_input)[:, 1]

        output = []
        for pred, prob in zip(predictions, probabilities):
            output.append({
                "loan_status_prediction": int(pred),
                "probability_of_default": round(float(prob), 4)
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)