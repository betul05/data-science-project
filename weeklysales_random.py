# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:\\Users\\Hp\\Downloads\\walmart-sales-dataset-of-45stores.csv")
# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

print(df.corr())

# Feature Engineering: Create 'Season' feature based on the month
df["Month"] = df["Date"].dt.month
df["Season"] = df["Month"].apply(lambda x: 
    "Winter" if x in [12, 1, 2] else 
    "Spring" if x in [3, 4, 5] else 
    "Summer" if x in [6, 7, 8] else 
    "Fall"
)

# Add a binary feature for the Christmas week (after Dec 20)
df["Christmas_Week"] = df["Date"].apply(lambda x: 1 if x.month == 12 and x.day >= 20 else 0)

# Add a binary feature for the Thanksgiving
df["Thanksgiving_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-11-21") <= x <= pd.to_datetime("2010-11-27") or
    pd.to_datetime("2011-11-20") <= x <= pd.to_datetime("2011-11-26") or
    pd.to_datetime("2012-11-18") <= x <= pd.to_datetime("2012-11-24") else 0)

# Add a binary feature for the Easter
df["Easter_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-03-29") <= x <= pd.to_datetime("2010-04-04") or
    pd.to_datetime("2011-04-18") <= x <= pd.to_datetime("2011-04-24") or
    pd.to_datetime("2012-04-02") <= x <= pd.to_datetime("2012-04-08") else 0)

# Add a binary feature for the Labor Day
df["Labor_Day_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-08-30") <= x <= pd.to_datetime("2010-09-05") or
    pd.to_datetime("2011-08-29") <= x <= pd.to_datetime("2011-09-04") or
    pd.to_datetime("2012-08-27") <= x <= pd.to_datetime("2012-09-02") else 0)

# Add a binary feature for the Super Bowl Week
df["Super_Bowl_Week"] = df["Date"].apply(lambda x: 1 if pd.to_datetime("2010-02-01") <= x <= pd.to_datetime("2010-02-07") or
    pd.to_datetime("2011-01-31") <= x <= pd.to_datetime("2011-02-06") or
    pd.to_datetime("2012-01-30") <= x <= pd.to_datetime("2012-02-05") else 0)

# Outlier Handling: Remove outliers using Z-score method
from scipy.stats import zscore
z_scores = np.abs(zscore(df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]]))
df = df[(z_scores < 3).all(axis=1)]  # Keep only rows where all z-scores < 3

# One-hot encode the 'Season' feature
df = pd.get_dummies(df, columns=["Season"], drop_first=False)  # Drop first to avoid dummy variable trap

print("\n")
print(df.columns)

# Define features (X) and target (y)
# Data Elimination: Drop 'Store' feature
X = df[["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Christmas_Week", "Thanksgiving_Week", "Easter_Week",
        "Labor_Day_Week","Super_Bowl_Week", "Season_Fall", "Season_Spring", "Season_Summer", "Season_Winter"]]
y = df["Weekly_Sales"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
def train_rf_model(X_train, y_train):
    """Trains a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Results:")
print(f"MAE: {mae_rf:.2f}")
print(f"MSE: {mse_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.4f}")

# Analyze feature importance
feature_importance_rf = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance_rf)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf, palette='viridis')
plt.title('Feature Importance - Random Forest Regressor')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('Actual vs Predicted - Random Forest Regressor')
plt.show()

# Model performance metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae_rf, mse_rf, rmse_rf, r2_rf]
})
print("\nModel Performance Metrics:")
print(metrics_df)

# Calculating store-based average sales
store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)

# Top 5 stores with highest average sales
print("\nTop 5 Stores with Highest Average Sales:")
print(store_sales.head(5))

# Bottom 5 stores with lowest average sales
print("\nBottom 5 Stores with Lowest Average Sales:")
print(store_sales.tail(5))

# Visualization of top 10 performing stores
plt.figure(figsize=(12,6))
store_sales.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Stores with Highest Average Sales RANDOM FOREST')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.show()

# Visualization of bottom 10 performing stores
plt.figure(figsize=(12,6))
store_sales.tail(10).plot(kind='bar', color='red')
plt.title('Bottom 10 Stores with Lowest Average Sales RANDOM FOREST')
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.show()
