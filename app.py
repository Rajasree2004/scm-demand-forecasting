from flask import Flask, request, jsonify, render_template

import joblib
import numpy as np
import pandas as pd
import os
import json
        
# from flask import Flask, request, jsonify, session, redirect, url_for
# from flask_pymongo import PyMongo
# from bcrypt import hashpw, checkpw, gensalt

# app = Flask(__name__)
# # app.config["MONGO_URI"] = "mongodb://localhost:27017/db-ml"
# # app.secret_key = "rajasree"  # Change this to a secure secret key
# # mongo = PyMongo(app)

# # # @app.route('/register', methods=['POST'])
# # # def register():
# # #     data = request.get_json()
# # #     username = data.get('username')
# # #     password = data.get('password')

# # #     hashed_password = hashpw(password.encode('utf-8'), gensalt())
# # #     user = {'username': username, 'password': hashed_password}

# # #     mongo.db.users.insert_one(user)
# # #     return jsonify({'message': 'User registered successfully'})

# # @app.route('/login', methods=['POST'])
# # def login():
# #     data = request.get_json()
# #     username = data.get('username')
# #     password = data.get('password')

# #     user = mongo.db.users.find_one({'username': username})

# #     if user and checkpw(password.encode('utf-8'), user['password']):
# #         session['username'] = username
# #         return jsonify({'message': 'Login successful'})

# #     return jsonify({'message': 'Invalid credentials'}), 401

# # @app.route('/logout')
# # def logout():
# #     session.clear()
# #     return jsonify({'message': 'Logout successful'})



# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')


# # Define endpoint for prediction
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get JSON data from request
    
#     predicts = []
#     data = request.form
#     # Convert data into numpy array and reshape to 2D array with one row
#     data = np.array([data['month'], data['budget']]).reshape(1, -1)

#     # # Convert data to dataframe
#     algos = ['Gradient Boosting','Random Forest','Linear Regression','Neural Network']
#     df = pd.DataFrame(data, columns=['overdue_sum', 'prod_limit'])

#     # Make prediction using model
#     model = joblib.load("output_notebook.pkl")     
  
#     prediction = model.predict(df)
#     response = {"prediction": prediction.tolist()}
#     predicts.append(response)
    
    
#     # model = joblib.load("rf_model.pkl")
#     # prediction = model.predict(df)
#     # response2 = {"prediction": prediction.tolist()}
#     # print(response2)
#     # predicts.append(response2)
    
    
#     # model = joblib.load("Linear.pkl")
#     # prediction = model.predict(df)
#     # response2 = {"prediction": prediction.tolist()}
#     # print(response2)
#     # predicts.append(response2)
    
#     # model = joblib.load("Neural.pkl")
#     # prediction = model.predict(df)
#     # response2 = {"prediction": prediction.tolist()}
#     # print(response2)
#     # predicts.append(response2)

    
#     return render_template('predict.html',data = predicts,algo=algos)


# if __name__ == "__main__":
#     app.run(debug=True)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template, jsonify
import joblib

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('output_notebook.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    features = np.array([float(data['ovd_sum']), float(data['prod_limit'])]).reshape(1,-1)
    print(data['ovd_sum'])
    print(data['prod_limit'])
    df = pd.DataFrame(features, columns=['overdue_sum', 'prod_limit'])
    prediction = model.predict(df)
    response = {
        "prediction":prediction.tolist()
    }
    return response

if __name__ == '__main__':
    app.run(debug=True)





































