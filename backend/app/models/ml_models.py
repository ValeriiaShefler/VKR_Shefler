import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from ..models.isa_core import standard_atmosphere

def generate_training_data(h_min=0, h_max=20000, step=100):
    heights = np.arange(h_min, h_max + step, step)
    rows = []
    for h in heights:
        T, P, rho, a = standard_atmosphere(h)
        rows.append([h, T, P, rho, a])
    return pd.DataFrame(rows, columns=['h', 'T', 'P', 'ρ', 'a'])

def train_mlp_models():
    df = generate_training_data()
    X = df[['h']].values
    y_T = df['T'].values
    y_P = df['P'].values
    y_ρ = df['ρ'].values
    y_a = df['a'].values
    
    # Нормализация входов
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_poly)
    
    # Нормализация выходов (особенно для P)
    scaler_P = StandardScaler()
    y_P_scaled = scaler_P.fit_transform(y_P.reshape(-1, 1)).ravel()
    
    # Остальные выходы не нормализуем (они и так в хороших диапазонах)
    model_T = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', max_iter=3000, random_state=42)
    model_P = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', max_iter=3000, random_state=42)
    model_ρ = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', max_iter=3000, random_state=42)
    model_a = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', max_iter=3000, random_state=42)
    
    print("Обучение модели T...")
    model_T.fit(X_scaled, y_T)
    print("Обучение модели P (с нормализацией)...")
    model_P.fit(X_scaled, y_P_scaled)
    print("Обучение модели ρ...")
    model_ρ.fit(X_scaled, y_ρ)
    print("Обучение модели a...")
    model_a.fit(X_scaled, y_a)
    
    models = {
        'T': model_T,
        'P': model_P,
        'ρ': model_ρ,
        'a': model_a,
        'scaler_X': scaler_X,
        'scaler_P': scaler_P,
        'poly': poly
    }
    
    test_data = (X_scaled, y_T, y_P, y_ρ, y_a)
    return models, test_data