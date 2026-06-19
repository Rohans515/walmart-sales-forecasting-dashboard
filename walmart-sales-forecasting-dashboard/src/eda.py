import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def run_eda():
    print("\n--- STARTING EDA ---\n")

    df = pd.read_csv("data/walmart_sales.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nSummary Statistics:")
    print(df.describe())

    # Sales distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['total_sale'], kde=True)
    plt.title("Sales Distribution")
    plt.savefig("sales_distribution.png")
    plt.close()

    # Daily trend
    daily_sales = df.groupby("order_date")['total_sale'].sum()

    plt.figure(figsize=(12,6))
    plt.plot(daily_sales)
    plt.title("Daily Sales Trend")
    plt.savefig("daily_sales_trend.png")
    plt.close()

    # Time-series decomposition
    decomposition = seasonal_decompose(daily_sales, model="additive")
    decomposition.plot()
    plt.savefig("time_series_decomposition.png")
    plt.close()

    # Category-wise sales
    plt.figure(figsize=(10,5))
    sns.barplot(x="product_category", y="total_sale", data=df)
    plt.xticks(rotation=45)
    plt.title("Category-wise Sales")
    plt.savefig("category_sales.png")
    plt.close()

    # Store-wise sales
    store_sales = df.groupby("store_location")['total_sale'].sum()

    plt.figure(figsize=(10,5))
    store_sales.plot(kind='bar')
    plt.xticks(rotation=45)
    plt.title("Store-wise Sales")
    plt.savefig("store_sales.png")
    plt.close()

    print("\nEDA Completed. Images saved in project folder.\n")
