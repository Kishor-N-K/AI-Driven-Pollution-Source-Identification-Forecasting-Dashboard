import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# Load CPCB historical data
df = pd.read_csv("delhi_cpcb_historical.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Feature Engineering
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

df["lag_1"] = df["aqi"].shift(1)
df["lag_2"] = df["aqi"].shift(2)
df["lag_3"] = df["aqi"].shift(3)

df["rolling_mean_3"] = df["aqi"].rolling(3).mean()
df["rolling_mean_6"] = df["aqi"].rolling(6).mean()

df = df.dropna()

features = [
    "hour", "day_of_week", "month",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_6"
]

X = df[features]
y = df["aqi"]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)

model.fit(X, y)

joblib.dump(model, "aqi_model.pkl")

print("Model trained and saved.")
