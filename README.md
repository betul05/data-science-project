# Walmart Weekly Sales Forecasting
This project analyzes and predicts weekly Walmart sales using **Random Forest** and **XGBoost** regression models.  
The dataset contains historical sales, holiday flags, and economic indicators for 45 stores.

---

## Project Overview
The goal of this project is to identify the most important factors influencing weekly sales and to compare the performance of two different machine learning models.

---

### Models Used
- Random Forest Regressor
- XGBoost Regressor

### Feature Engineering
- Date conversion and extraction of month/season
- Binary flags for major US holidays (Christmas, Thanksgiving, Easter, Labor Day, Super Bowl)
- Outlier removal using Z-Score
- One-hot encoding for categorical features

---

## Technologies Used
- Python  
- Pandas, NumPy, Seaborn, Matplotlib  
- Scikit-learn  
- XGBoost  

---

## Visualizations
- Feature importance plots  
- Actual vs predicted sales  
- Top and bottom performing stores  

---

## Dataset
Dataset: [Walmart Store Sales Forecasting](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

