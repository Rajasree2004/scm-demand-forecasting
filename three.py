import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
# Load data
data = pd.read_csv('data.csv')

# Preprocess data
X = data.drop(['sales'], axis=1)
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = [LinearRegression(), MLPRegressor(hidden_layer_sizes=(100,100))]
model_names = ['Linear Regression', 'Neural Network']
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(model_names[i] + " Mean Squared Error:", mse)
    joblib.dump(model, f'{model_names[i].split()[0]}.pkl')