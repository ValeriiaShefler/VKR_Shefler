"""
Роутер для работы с ИИ: обучение, предсказание, статус.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np

from ..services.trainer import (
    train, predict_point, list_trained, clear_cache,
    get_default_hyperparams, DEFAULT_HYPERPARAMS,
)
from ..data_loader import load_dataset, get_dataset_info


router = APIRouter(prefix="/ai", tags=["ИИ"])


# ---------- Pydantic-схемы ----------
class TrainRequest(BaseModel):
    dataset: str
    method: str
    hyperparams: Optional[Dict[str, Any]] = None
    test_size: float = 0.2


class TrainResponse(BaseModel):
    cache_key: str
    dataset: str
    method: str
    train_time_s: float
    n_train: int
    n_test: int
    metrics: Dict[str, float]
    hyperparams: Dict[str, Any]


class PredictRequest(BaseModel):
    cache_key: str
    # Для 1D — только h. Для 2D+ — остальные параметры.
    h: float
    latitude_deg: Optional[float] = None
    season: Optional[str] = None
    w: Optional[float] = None


class PredictRangeRequest(BaseModel):
    cache_key: str
    h_start: float
    h_end: float
    h_step: float = 100.0
    latitude_deg: Optional[float] = None
    season: Optional[str] = None
    w: Optional[float] = None


# ---------- Эндпоинты ----------
@router.get("/default_hyperparams/{method}")
async def default_hyperparams(method: str):
    """Гиперпараметры по умолчанию для метода."""
    try:
        return get_default_hyperparams(method)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    """Обучает модель на датасете."""
    try:
        info = train(req.dataset, req.method,
                      hyperparams=req.hyperparams,
                      test_size=req.test_size)
        return TrainResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list_trained")
async def list_trained_models():
    """Список всех обученных моделей."""
    return list_trained()


@router.post("/clear_cache")
async def clear_models_cache():
    """Очищает кэш обученных моделей."""
    clear_cache()
    return {"status": "cache cleared"}


def _build_input_vector(req_data: dict, dataset_info: dict) -> np.ndarray:
    """Собирает входной вектор для модели в зависимости от датасета."""
    features = dataset_info["features"]
    row = []

    for f in features:
        if f == "h":
            row.append(req_data["h"])
        elif f == "latitude_deg":
            if req_data.get("latitude_deg") is None:
                raise HTTPException(status_code=400,
                                    detail="Нужен параметр latitude_deg")
            row.append(req_data["latitude_deg"])
        elif f == "w":
            w = req_data.get("w")
            if w is None:
                w = 0.5
            row.append(w)
        elif f.startswith("season_"):
            season = req_data.get("season", "annual")
            row.append(1.0 if f == f"season_{season}" else 0.0)
    return np.array([row])


@router.post("/predict")
async def predict(req: PredictRequest):
    """Предсказание для одной точки."""
    try:
        # Определяем датасет по cache_key
        cache_key = req.cache_key
        dataset = cache_key.split("|")[0]
        info = get_dataset_info(dataset)

        X = _build_input_vector(req.dict(), info)
        result = predict_point(cache_key, X)
        return {"input": req.dict(), "output": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict_range")
async def predict_range(req: PredictRangeRequest):
    """Предсказание для интервала высот."""
    try:
        cache_key = req.cache_key
        dataset = cache_key.split("|")[0]
        info = get_dataset_info(dataset)

        heights = np.arange(req.h_start, req.h_end + req.h_step, req.h_step)
        rows = []
        for h in heights:
            req_data = {
                "h": float(h),
                "latitude_deg": req.latitude_deg,
                "season": req.season,
                "w": req.w,
            }
            X = _build_input_vector(req_data, info)
            result = predict_point(cache_key, X)
            rows.append({
                "h": float(h),
                "T": result["T"],
                "P": result["P"],
                "rho": result["rho"],
                "a": result["a"],
            })

        return {"n_points": len(rows), "data": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))