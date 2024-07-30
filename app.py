from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load('best_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "API de prédiction des prix des maisons"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
