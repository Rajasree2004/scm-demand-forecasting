# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import joblib
# Load data
data = pd.read_csv('data.csv', index_col=0)

# Visualize data
plt.plot(data)
plt.title('Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# Train ARIMA model
model = ARIMA(data, order=(2, 1, 2))
model_fit = model.fit(disp=0)

joblib.dump(model_fit, 'arima.pkl')
# Forecast sales
forecast = model_fit.forecast(steps=6)[0]

