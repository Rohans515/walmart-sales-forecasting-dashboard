import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_forecast():
    print("\n--- STARTING FORECASTING ---\n")

    df = pd.read_csv("data/walmart_sales.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])

    ts = df.groupby("order_date")['total_sale'].sum()

    # Train ARIMA model
    model = ARIMA(ts, order=(5,1,2))
    model_fit = model.fit()

    # Forecast next 30 days
    forecast = model_fit.forecast(30)

    # Plot forecast vs actual
    plt.figure(figsize=(12,6))
    plt.plot(ts, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="dashed")
    plt.legend()
    plt.title("30 Day Sales Forecast")
    plt.savefig("forecast_plot.png")
    plt.close()

    # Accuracy evaluation for last 30 days
    y_true = ts[-30:]
    y_pred = model_fit.predict(start=len(ts)-30, end=len(ts)-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    print("\nForecast Completed. Output saved as forecast_plot.png\n")
