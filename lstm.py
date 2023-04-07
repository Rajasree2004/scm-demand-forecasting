import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

# Load data
data = pd.read_csv("data.csv")

# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into features and target
X = []
y = []
n_future = 1  # number of months to predict
n_past = 12  # number of months to use as input
for i in range(n_past, len(data_scaled) - n_future + 1):
    X.append(data_scaled[i - n_past:i, 1:])
    y.append(data_scaled[i + n_future - 1:i + n_future, 0])
X, y = np.array(X), np.array(y)

# Train/test split
split = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Save model as h5 file
model.save('lstm_model.h5')

# Load saved model
loaded_model = tf.keras.models.load_model('lstm_model.h5')

# Convert h5 file to pkl file
with open('lstm_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

# Load data
data = pd.read_csv("data.csv")

# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into features and target
X = []
y = []
n_future = 1  # number of months to predict
n_past = 12  # number of months to use as input
for i in range(n_past, len(data_scaled) - n_future + 1):
    X.append(data_scaled[i - n_past:i, 1:])
    y.append(data_scaled[i + n_future - 1:i + n_future, 0])
X, y = np.array(X), np.array(y)

# Train/test split
split = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Save model as h5 file
model.save('lstm_model.h5')

# Load saved model
loaded_model = tf.keras.models.load_model('lstm_model.h5')

# Convert h5 file to pkl file
with open('lstm_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)
