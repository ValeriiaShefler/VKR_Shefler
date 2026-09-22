# experiments/diagnose_gpr_3d.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "datasets", "3d.csv"))
season_dummies = pd.get_dummies(df["season"], prefix="season")
df = pd.concat([df, season_dummies], axis=1)

feature_cols = ["h", "latitude_deg",
                "season_winter", "season_summer", "season_annual"]
X = df[feature_cols].values
y = df[["T", "P", "rho", "a"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_tr_s = scaler_X.fit_transform(X_train)
X_te_s = scaler_X.transform(X_test)

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6)

for i, param in enumerate(["T", "P", "rho", "a"]):
    # Без log-transform
    m = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    m.fit(X_tr_s, y_train[:, i])
    pred = m.predict(X_te_s)
    mae = mean_absolute_error(y_test[:, i], pred)
    r2 = r2_score(y_test[:, i], pred)
    print(f"{param}: MAE={mae:.4f}, R²={r2:.4f}")