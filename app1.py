from flask import Flask, request, session, render_template, redirect, jsonify
import joblib
import numpy as np
import requests
model = joblib.load("gbm_model.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        # Convert data into numpy array
        data = np.array([data['month'], data['budget'], data['competitors_price'], data['holidays'], data['sales']])
        # Make prediction using loaded model
        prediction = model.predict(data)
        # BASE = "http://127.0.0.1:5000/"
        # # Set request's header.
        # headers = {"Content-Type": "application/json; charset=utf-8"}
        # # Set data.
        # data = {"likes": 10}
        # # 
        # response = requests.post(BASE + 'predict', headers=headers, json=data)

        # print("Status Code: ", response.status_code)
        # print("JSON Response: ", response.json())
        
        return jsonify(list(prediction))


if(__name__ == '__main__'):
    app.run(debug= True)