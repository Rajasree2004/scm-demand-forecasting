import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data from CSV file
data = pd.read_csv("data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('sales', axis=1), data['sales'], test_size=0.2, random_state=42)

# Train a Gradient Boosting model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions on test set and calculate RMSE
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

# Save model using joblib
joblib.dump(model, "gbm_model.pkl")
