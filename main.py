import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta

start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(50)]

# Generate follower counts with a increasing trend and random variation
initial_followers = 1000
follower_counts = [initial_followers + i * 50 + np.random.randint(-200, 200) for i in range(50)]


# Create a pandas DataFrame
data = pd.DataFrame({'Date': pd.to_datetime(dates), 'FollowerCount': follower_counts})
data.set_index('Date', inplace=True)

# Fit a SARIMA model
sarima_model = SARIMAX(data['FollowerCount'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Calculate model accuracy on existing data
actual_values = data['FollowerCount']
predicted_values = sarima_fit.fittedvalues
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)

# Forecast future follower counts
future_dates = [data.index[-1] + pd.DateOffset(days=i) for i in range(1, 31)]
forecast = sarima_fit.get_forecast(steps=30)
forecast_index = pd.to_datetime(future_dates)
forecast_values = forecast.predicted_mean.values

# Create a DataFrame with forecasted data
forecast_df = pd.DataFrame({'Date': forecast_index, 'FollowerCount': forecast_values})
forecast_df.set_index('Date', inplace=True)

# Combine actual and forecasted data
combined_data = data.append(forecast_df)

# Print model accuracy
print(f"Root Mean Squared Error (RMSE) on existing data: {rmse:.2f}")

# Print the combined data
print(combined_data)

# Plot the data using Plotly
fig = px.line(combined_data, x=combined_data.index, y='FollowerCount', title='Follower Count Forecast using SARIMA')
fig.show()
