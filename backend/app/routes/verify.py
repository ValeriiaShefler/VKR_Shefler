"""
Роутер для верификации: метрики MAE/RMSE/R².
"""
from fastapi import APIRouter, HTTPException
from ..services.trainer import _MODELS_CACHE


router = APIRouter(prefix="/verify", tags=["Верификация"])


@router.get("/{cache_key:path}")
async def get_verification(cache_key: str):
    """Возвращает метрики обученной модели по её cache_key."""
    if cache_key not in _MODELS_CACHE:
        raise HTTPException(status_code=404,
                            detail="Модель не обучена. Сначала /ai/train.")
    info = _MODELS_CACHE[cache_key]["info"]
    metrics = info["metrics"]

    # Формируем таблицу по параметрам
    rows = []
    for param in ["T", "P", "rho", "a"]:
        rows.append({
            "parameter": param,
            "mae": metrics.get(f"MAE_{param}"),
            "rmse": metrics.get(f"RMSE_{param}"),
            "r2": metrics.get(f"R2_{param}"),
        })
    return {
        "dataset": info["dataset"],
        "method": info["method"],
        "train_time_s": info["train_time_s"],
        "metrics_by_param": rows,
        "metrics_avg": {
            "MAE": metrics["MAE_avg"],
            "RMSE": metrics["RMSE_avg"],
            "R2": metrics["R2_avg"],
        },
    }


@router.get("/")
async def list_verifications():
    """Список всех обученных моделей с метриками."""
    from ..services.trainer import list_trained
    return list_trained()