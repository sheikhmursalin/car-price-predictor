"""
Train RandomForest model on data/car_data.csv
Run once:  python train_model.py
Creates:   models/car_price_model.pkl  +  models/meta.json
"""

import pandas as pd, numpy as np, os, json, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CURRENT_YEAR = 2025
DATA_PATH    = "data/car_data.csv"
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- 1. Load & Clean --------
df = pd.read_csv(DATA_PATH)

df = df[~df["Price"].str.contains("Ask", case=False)]
df["Price"] = df["Price"].str.replace(",", "").astype(int)
df["kms_driven"] = (
    df["kms_driven"].str.replace(",", "")
                    .str.extract(r"(\d+)")
                    .fillna("0").astype(int)
)
df = df[df["year"].str.isnumeric()]
df["year"] = df["year"].astype(int)
df = df.dropna(subset=["fuel_type"])
df["fuel_type"] = df["fuel_type"].str.strip()

df["car_age"] = CURRENT_YEAR - df["year"]

features = ["company", "fuel_type", "kms_driven", "car_age"]
target   = "Price"
X, y     = df[features], df[target]

# -------- 2. Preprocess + Train --------
cat_cols = ["company", "fuel_type"]
num_cols = ["kms_driven", "car_age"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("prep", pre),
    ("rf",  RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(f"Model trained | RMSE: ₹{rmse:,.0f}")

# -------- 3. Save Artefacts --------
joblib.dump(model, f"{MODEL_DIR}/car_price_model.pkl")

meta = {
    "companies": sorted(df["company"].unique()),
    "fuel_types": sorted(df["fuel_type"].unique()),
    "trained_year": CURRENT_YEAR
}
json.dump(meta, open(f"{MODEL_DIR}/meta.json", "w"), indent=2)
print(" Saved model & meta to models/ ")
