from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy, session
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)




# app.config['SECRET_KEY'] = 'helooooo'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.sqlite3'

# class predicts(db.Model):
#     _id = db.Column("id",db.Integer, primary_key=True)
#     num = db.Column("num",db.String(10), nullable=False)
#     message = db.Column(db.String(1200), nullable=False)

#     def __init__(self,num,message):
#         self.message = message
#         self.num = num
        
        

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Define endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from request
    predicts = []
    data = request.form
    # Convert data into numpy array and reshape to 2D array with one row
    data = np.array([data['month'], data['budget'], data['competitors_price'], data['holidays']]).reshape(1, -1)

    # Convert data to dataframe
    algos = ['Gradient Boosting','Random Forest','Linear Regression','Neural Network']
    df = pd.DataFrame(data, columns=['month', 'marketing_budget', 'competitors_price', 'holidays'])

    # Make prediction using model
    model = joblib.load("gbm_model.pkl")
    prediction = model.predict(df)
    response = {"prediction": prediction.tolist()}
    predicts.append(response)
    
    
    model = joblib.load("rf_model.pkl")
    prediction = model.predict(df)
    response2 = {"prediction": prediction.tolist()}
    print(response2)
    predicts.append(response2)
    
    
    model = joblib.load("Linear.pkl")
    prediction = model.predict(df)
    response2 = {"prediction": prediction.tolist()}
    print(response2)
    predicts.append(response2)
    
    model = joblib.load("Neural.pkl")
    prediction = model.predict(df)
    response2 = {"prediction": prediction.tolist()}
    print(response2)
    predicts.append(response2)

    
    return render_template('predict.html',data = predicts, algos = algos)


if __name__ == "__main__":
    app.run(debug=True)
