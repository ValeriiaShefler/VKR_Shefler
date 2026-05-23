from fastapi import APIRouter, HTTPException
from ..models.ml_models import train_mlp_models
from ..schemas.responses import AIPrediction
from ..config import MODELS_DIR
import pickle
import os

router = APIRouter(prefix="/ai", tags=["AI аппроксимация"])

training_status = {"status": "not_started", "message": ""}

def load_models():
    models = {}
    for name in ['T', 'P', 'ρ', 'a', 'scaler_X', 'scaler_P', 'poly']:
        file_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            models[name] = pickle.load(f)
    return models

@router.post("/train")
async def train_model():
    global training_status
    try:
        training_status = {"status": "in_progress", "message": "🔄 Обучение началось..."}
        
        models, _ = train_mlp_models()
        
        for name, model in models.items():
            with open(os.path.join(MODELS_DIR, f"{name}_model.pkl"), "wb") as f:
                pickle.dump(model, f)
        
        training_status = {
            "status": "completed",
            "message": f"✅ Обучение завершено! Модели в: {MODELS_DIR}"
        }
        return training_status
    except Exception as e:
        training_status = {"status": "failed", "message": f"❌ Ошибка: {str(e)}"}
        return training_status

@router.get("/status")
async def get_training_status():
    return training_status

@router.get("/predict", response_model=AIPrediction)
async def predict(height: float):
    models = load_models()
    if models is None:
        raise HTTPException(status_code=404, detail="Модель не обучена")
    
    if height < 0 or height > 20000:
        raise HTTPException(status_code=400, detail="Высота должна быть 0..20000 м")
    
    X_poly = models['poly'].transform([[height]])
    X_scaled = models['scaler_X'].transform(X_poly)
    
    T = float(models['T'].predict(X_scaled)[0])
    P_scaled = models['P'].predict(X_scaled)[0]
    P = float(models['scaler_P'].inverse_transform([[P_scaled]])[0][0])
    rho = float(models['ρ'].predict(X_scaled)[0])
    a = float(models['a'].predict(X_scaled)[0])
    
    return AIPrediction(
        height_m=height,
        temperature_K=round(T, 2),
        pressure_Pa=round(P, 2),
        density_kg_m3=round(rho, 4),
        speed_of_sound_m_s=round(a, 2)
    )