# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("C:\\Users\\Hp\\Downloads\\walmart-sales-dataset-of-45stores.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Feature Engineering: Create 'Season' feature
df["Month"] = df["Date"].dt.month
df["Season"] = df["Month"].apply(lambda x: 
    "Winter" if x in [12, 1, 2] else 
    "Spring" if x in [3, 4, 5] else 
    "Summer" if x in [6, 7, 8] else 
    "Fall"
)

# Add holiday-related binary features
df["Christmas_Week"] = df["Date"].apply(lambda x: 1 if x.month == 12 and x.day >= 20 else 0)
df["Thanksgiving_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-11-21") <= x <= pd.to_datetime("2010-11-27") or
    pd.to_datetime("2011-11-20") <= x <= pd.to_datetime("2011-11-26") or
    pd.to_datetime("2012-11-18") <= x <= pd.to_datetime("2012-11-24") else 0)
df["Easter_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-03-29") <= x <= pd.to_datetime("2010-04-04") or
    pd.to_datetime("2011-04-18") <= x <= pd.to_datetime("2011-04-24") or
    pd.to_datetime("2012-04-02") <= x <= pd.to_datetime("2012-04-08") else 0)
df["Labor_Day_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-08-30") <= x <= pd.to_datetime("2010-09-05") or
    pd.to_datetime("2011-08-29") <= x <= pd.to_datetime("2011-09-04") or
    pd.to_datetime("2012-08-27") <= x <= pd.to_datetime("2012-09-02") else 0)
df["Super_Bowl_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-02-01") <= x <= pd.to_datetime("2010-02-07") or
    pd.to_datetime("2011-01-31") <= x <= pd.to_datetime("2011-02-06") or
    pd.to_datetime("2012-01-30") <= x <= pd.to_datetime("2012-02-05") else 0)

# Outlier removal using Z-score
from scipy.stats import zscore
z_scores = np.abs(zscore(df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]]))
df = df[(z_scores < 3).all(axis=1)]

# One-hot encode season
df = pd.get_dummies(df, columns=["Season"], drop_first=False)

# Define features and target
X = df[["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Christmas_Week", "Thanksgiving_Week", "Easter_Week", "Labor_Day_Week",
        "Super_Bowl_Week", "Season_Fall", "Season_Spring", "Season_Summer", "Season_Winter"]]
y = df["Weekly_Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# XBoost model function
def train_xgb_model(X_train, y_train):
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Regressor Results:")
print(f"MAE: {mae_xgb:.2f}")
print(f"MSE: {mse_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R² Score: {r2_xgb:.4f}")

# Feature importance
feature_importance_xgb = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (XGBoost):")
print(feature_importance_xgb)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_xgb, palette='magma')
plt.title('Feature Importance - XGBoost Regressor')
plt.show()

# Actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_xgb)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('Actual vs Predicted - XGBoost Regressor')
plt.show()

# Performance metrics table
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae_xgb, mse_xgb, rmse_xgb, r2_xgb]
})
print("\nModel Performance Metrics:")
print(metrics_df)

# Calculate average weekly sales by store
store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)

# Display top 5 stores
print("\nTop 5 Stores with Highest Average Sales:")
print(store_sales.head(5))

# Display bottom 5 stores
print("\nBottom 5 Stores with Lowest Average Sales:")
print(store_sales.tail(5))

# Visualization: Top 10 stores
plt.figure(figsize=(12,6))
store_sales.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Stores with Highest Average Sales XGBOOST')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualization: Bottom 10 stores
plt.figure(figsize=(12,6))
store_sales.tail(10).plot(kind='bar', color='red')
plt.title('Bottom 10 Stores with Lowest Average Sales XGBOOST')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

