import pandas as pd
from fbprophet import Prophet

# Load data into a pandas DataFrame
df = pd.read_csv("data.csv")

# Convert month column to datetime format
df["month"] = pd.to_datetime(df["month"], format="%m")

# Rename columns to fit Prophet's naming convention
df = df.rename(columns={"month": "ds", "sales": "y"})

# Create a Prophet model
model = Prophet()

# Add holiday effects to the model
model.add_country_holidays(country_name='US')

# Fit the model to the data
model.fit(df)

# Make a prediction for the next 12 months
future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

# Print the forecast
print(forecast[["ds", "yhat"]].tail(12))
