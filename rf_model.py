import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv('data.csv')

# Split data into features and target
X = data.drop('sales', axis=1)
y = data['sales']

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X, y)

# Make predictions on the same data to evaluate model
y_pred = rf_model.predict(X)

# Calculate RMSE on predictions
rmse = mean_squared_error(y, y_pred, squared=False)

print('RMSE:', rmse)

# Save model
joblib.dump(rf_model, 'rf_model.pkl')
