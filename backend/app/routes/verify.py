from fastapi import APIRouter, HTTPException
from ..services.verification import compute_metrics
from ..schemas.responses import VerificationMetrics
from ..config import MODELS_DIR
from ..models.ml_models import generate_training_data
import pickle
import os

router = APIRouter(prefix="/verify", tags=["Верификация и валидация"])

def load_models():
    models = {}
    for name in ['T', 'P', 'ρ', 'a', 'scaler_X', 'scaler_P', 'poly']:
        file_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            models[name] = pickle.load(f)
    return models

@router.get("/", response_model=list[VerificationMetrics])
async def get_verification():
    models = load_models()
    if models is None:
        raise HTTPException(status_code=404, detail="Модель не обучена")
    
    # Генерируем данные для проверки
    df = generate_training_data()
    X = df[['h']].values
    y_T = df['T'].values
    y_P = df['P'].values
    y_ρ = df['ρ'].values
    y_a = df['a'].values
    
    # Полиномиальные признаки + нормализация
    X_poly = models['poly'].transform(X)
    X_scaled = models['scaler_X'].transform(X_poly)
    
    # Предсказания
    yT_pred = models['T'].predict(X_scaled)
    yP_pred_scaled = models['P'].predict(X_scaled)
    yP_pred = models['scaler_P'].inverse_transform(yP_pred_scaled.reshape(-1, 1)).ravel()
    yρ_pred = models['ρ'].predict(X_scaled)
    ya_pred = models['a'].predict(X_scaled)
    
    results = []
    for name, y_true, y_pred in zip(['T', 'P', 'ρ', 'a'], [y_T, y_P, y_ρ, y_a], [yT_pred, yP_pred, yρ_pred, ya_pred]):
        mae, rmse, r2 = compute_metrics(y_true, y_pred)
        results.append(VerificationMetrics(parameter=name, mae=mae, rmse=rmse, r2=r2))
    
    return results