"""
Роутер для эталонных данных (ГОСТ Р 70469-2026 / ISO 5878).
Используется для построения графиков и сравнения.
"""
from fastapi import APIRouter, HTTPException
from ..data_loader import load_dataset, get_dataset_info, TARGETS
import numpy as np


router = APIRouter(prefix="/reference", tags=["Эталон"])


@router.get("/{dataset}")
async def get_reference(dataset: str, param: str = "T",
                         h_min: float = None, h_max: float = None,
                         latitude_deg: float = None,
                         season: str = "annual",
                         w: float = 0.5):
    """
    Возвращает эталонные значения параметра по высоте.

    Параметры:
        dataset      — 1d / 2d / 3d / 4d
        param        — T / P / rho / a
        h_min, h_max — диапазон высот (м). Если None — весь диапазон.
        latitude_deg, season, w — фиксированные значения остальных осей.
    """
    try:
        info = get_dataset_info(dataset)
        df = load_dataset(dataset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Фильтрация
    if h_min is not None:
        df = df[df["h"] >= h_min]
    if h_max is not None:
        df = df[df["h"] <= h_max]

    # Для 2D+ фильтруем по фиксированным значениям
    if "latitude_deg" in df.columns and latitude_deg is not None:
        df = df[np.isclose(df["latitude_deg"], latitude_deg, atol=1.0)]

    if "season" in df.columns and season is not None:
        df = df[df["season"] == season]

    if "w" in df.columns and w is not None:
        df = df[np.isclose(df["w"], w, atol=0.05)]

    if param not in TARGETS:
        raise HTTPException(status_code=400, detail=f"Параметр {param} не найден")

    # Сортируем по высоте
    df = df.sort_values("h")

    return {
        "dataset": dataset,
        "param": param,
        "standard": info["standard"],
        "n_points": len(df),
        "data": df[["h", param]].rename(columns={param: "value"}).to_dict(orient="records"),
    }