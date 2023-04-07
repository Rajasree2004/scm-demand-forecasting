from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Define endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from request
    data = request.form
    # Convert data into numpy array and reshape to 2D array with one row
    data = np.array([data['month'], data['budget'], data['competitors_price'], data['holidays']]).reshape(1, -1)

    # Convert data to dataframe
    df = pd.DataFrame(data, columns=['month', 'marketing_budget', 'competitors_price', 'holidays'])

    # Make prediction using model
    model = joblib.load("gbm_model.pkl")
    prediction = model.predict(df)

    # Return prediction as JSON response
    response = {"prediction": prediction.tolist()}
    print(response)
    
    model = joblib.load("rf_model.pkl")
    prediction = model.predict(df)

    # Return prediction as JSON response
    response2 = {"prediction": prediction.tolist()}
    print(response2)
    

    
    return render_template('predict.html',data = response,response2=response2 )


if __name__ == "__main__":
    app.run(debug=True)
