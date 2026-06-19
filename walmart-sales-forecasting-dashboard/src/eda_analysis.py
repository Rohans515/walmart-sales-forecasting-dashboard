
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("visualizations", exist_ok=True)

def save_plot(fname):
   
    path = os.path.join("visualizations", fname)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot saved at: {path}")

    plt.close()


df = pd.read_csv(r"D:\c vs code\walmart project\data\walmart_sales_extended 1.csv")



print("\n===== HEAD =====")
print(df.head())


print("\n===== INFO =====")
print(df.info())

# Summary statistics
print("\n===== DESCRIBE =====")
print(df.describe())

# Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Duplicate check
print("\n===== DUPLICATES =====")
print(df.duplicated().sum())



# Histogram
plt.figure(figsize=(7,5))
df['Total_Sales'].hist()
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
# earlier: plt.show()
save_plot("sales_distribution.png")

# Correlation heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Columns Only)")
save_plot("correlation_heatmap.png")


# Monthly sales line chart (if date column exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Sales'].sum()

    plt.figure(figsize=(8,5))
    monthly.plot(kind="line")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    save_plot("monthly_sales_trend.png")
    plt.show()


# ===== OUTLIER DETECTION USING BOXPLOTS =====

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Total_Sales'])
plt.title("Boxplot of Total Sales (Outlier Detection)")
save_plot("boxplot_total_sales.png")

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Unit_Price'])
plt.title("Boxplot of Unit Price (Outlier Detection)")
save_plot("boxplot_unit_price.png")

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Quantity'])
plt.title("Boxplot of Quantity (Outlier Detection)")
save_plot("boxplot_quantity.png")
# ===== OUTLIER DETECTION USING Z-SCORE =====

from scipy import stats
import numpy as np

numeric_df = df[['Total_Sales', 'Unit_Price', 'Quantity']]

z_scores = np.abs(stats.zscore(numeric_df))

outliers = np.where(z_scores > 3)
print("\n===== Z-SCORE OUTLIERS (values > 3 are outliers) =====")
print(outliers)
# ===== TIME SERIES PREPARATION =====

df['Order_Date'] = pd.to_datetime(df['Order_Date'])

df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Month_Name'] = df['Order_Date'].dt.strftime("%b")
df['Week'] = df['Order_Date'].dt.isocalendar().week
df['Date'] = df['Order_Date'].dt.date
# ===== DAILY SALES TREND =====

daily_sales = df.groupby('Order_Date')['Total_Sales'].sum()

plt.figure(figsize=(10,5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
save_plot("daily_sales_trend.png")

# ===== MONTHLY SALES TREND =====

monthly_sales = df.groupby(['Year', 'Month'])['Total_Sales'].sum()

monthly_sales.plot(figsize=(10,5))
plt.title("Monthly Sales Trend")
plt.xlabel("Year, Month")
plt.ylabel("Total Sales")
save_plot("monthly_sales_trend.png")

# ===== MOVING AVERAGE (7-day) =====

daily_sales_ma = daily_sales.rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(daily_sales.index, daily_sales.values, alpha=0.3, label="Actual Sales")
plt.plot(daily_sales.index, daily_sales_ma, label="7-Day Moving Average")
plt.title("7-Day Moving Average of Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
save_plot("moving_average_7day.png")


# ===== WEEKLY TREND =====

weekly_sales = df.groupby('Week')['Total_Sales'].sum()

plt.figure(figsize=(10,5))
plt.plot(weekly_sales.index, weekly_sales.values)
plt.title("Weekly Sales Trend")
plt.xlabel("Week Number")
plt.ylabel("Total Sales")
plt.show()
save_plot("weekly_sales_trend.png")

# Use daily sales for modeling
data = daily_sales.reset_index()
data.columns = ["Date", "Sales"]

# Create a numerical index for regression model
data["Day_Index"] = range(len(data))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Train-test split
X = data[["Day_Index"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict future 30 days
future_days = 30
last_day = data["Day_Index"].max()

future_indices = pd.DataFrame({"Day_Index": range(last_day+1, last_day+1+future_days)})

future_predictions = model.predict(future_indices)
plt.figure(figsize=(12,6))

# Actual sales
plt.plot(data["Date"], data["Sales"], label="Actual Sales")

# Forecasted sales
future_dates = pd.date_range(start=data["Date"].max(), periods=future_days+1, freq='D')[1:]
plt.plot(future_dates, future_predictions, label="Forecasted Sales")

plt.title("Linear Regression Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
save_plot("linear_regression_forecast.png")




# Convert date column to datetime (if not already)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Group daily sales
daily_sales = df.groupby("Order_Date")['Total_Sales'].sum().reset_index()

# Sort by date
daily_sales = daily_sales.sort_values("Order_Date")

print("\n===== DAILY SALES (HEAD) =====")
print(daily_sales.head())

from sklearn.linear_model import LinearRegression
import numpy as np

# Create time index
daily_sales['TimeIndex'] = np.arange(len(daily_sales))

# Train model
X = daily_sales[['TimeIndex']]
y = daily_sales['Total_Sales']

model_lr = LinearRegression()
model_lr.fit(X, y)

# Predict next 30 days
future_index = np.arange(len(daily_sales), len(daily_sales) + 30)
future_preds_lr = model_lr.predict(future_index.reshape(-1, 1))

# Create forecast dataframe
lr_forecast_df = pd.DataFrame({
    "Date": pd.date_range(start=daily_sales['Order_Date'].iloc[-1] + pd.Timedelta(days=1), periods=30),
    "Forecast_LR": future_preds_lr
})

print("\n===== LINEAR REGRESSION FORECAST (HEAD) =====")
print(lr_forecast_df.head())

# Plot
plt.figure(figsize=(10,6))
plt.plot(daily_sales['Order_Date'], daily_sales['Total_Sales'], label="Historical")
plt.plot(lr_forecast_df['Date'], lr_forecast_df['Forecast_LR'], label="LR Forecast")
plt.legend()
plt.title("Linear Regression – 30 Day Forecast")
plt.show()
save_plot("linear_regression_30day_forecast.png")

# Save results to CSV
lr_forecast_df.to_csv("visualizations/linear_regression_forecast.csv", index=False)
print("Saved: visualizations/linear_regression_forecast.csv")



from statsmodels.tsa.arima.model import ARIMA

# ARIMA model (1,1,1) – safe beginner model
arima_model = ARIMA(daily_sales['Total_Sales'], order=(1, 1, 1))
arima_fit = arima_model.fit()

# Forecast next 30 days
arima_forecast = arima_fit.forecast(steps=30)

# Build ARIMA dataframe
arima_forecast_df = pd.DataFrame({
    "Date": pd.date_range(start=daily_sales['Order_Date'].iloc[-1] + pd.Timedelta(days=1), periods=30),
    "Forecast_ARIMA": arima_forecast.values
})

print("\n===== ARIMA FORECAST (HEAD) =====")
print(arima_forecast_df.head())

# Plot
plt.figure(figsize=(10,6))
plt.plot(daily_sales['Order_Date'], daily_sales['Total_Sales'], label="Historical")
plt.plot(arima_forecast_df['Date'], arima_forecast_df['Forecast_ARIMA'], label="ARIMA Forecast")
plt.legend()
plt.title("ARIMA – 30 Day Forecast")

save_plot("arima_30day_forecast.png")

# Save to CSV
arima_forecast_df.to_csv("visualizations/arima_forecast.csv", index=False)
print("Saved: visualizations/arima_forecast.csv")





import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model_arima = ARIMA(data["Sales"], order=(5,1,2))
arima_result = model_arima.fit()

# Forecast next 30 days
arima_forecast = arima_result.forecast(steps=30)
# ======= PLOT ARIMA FORECAST =======

plt.figure(figsize=(12,6))

plt.plot(data["Date"], data["Sales"], label="Actual Sales")
plt.plot(future_dates, arima_forecast, label="ARIMA Forecast")

plt.title("ARIMA Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
save_plot("arima_forecast.png")




