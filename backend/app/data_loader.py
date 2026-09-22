"""
Загрузка и кэширование датасетов 1D–4D для MSA Studio.

Датасеты:
    1d — ГОСТ Р 70469-2026 (высота)
    2d — ISO 5878:2026 (высота + широта)
    3d — ISO 5878:2026 (высота + широта + сезон)
    4d — ISO 5878:2026 + влажность (высота + широта + сезон + w)

Все датасеты содержат целевые параметры: T, P, rho, a.
"""
import os
import pandas as pd
from functools import lru_cache


# ---------- Пути ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")


# ---------- Метаданные датасетов ----------
DATASET_META = {
    "1d": {
        "label": "1D — ГОСТ Р 70469-2026 (по высоте)",
        "features": ["h"],
        "feature_labels": {"h": "Высота, м"},
        "h_range": (0, 20000),
        "standard": "ГОСТ Р 70469-2026 / ISO 2533:2026",
        "description": "Только высота. Классическая модель МСА по ГОСТ Р 70469-2026.",
    },
    "2d": {
        "label": "2D — ISO 5878 (высота + широта)",
        "features": ["h", "latitude_deg"],
        "feature_labels": {
            "h": "Высота, м",
            "latitude_deg": "Широта, °",
        },
        "h_range": (0, 80000),
        "latitude_range": (15, 80),
        "standard": "ISO 5878:2026",
        "description": "Высота и широта. Сезон усреднён.",
    },
    "3d": {
        "label": "3D — ISO 5878 (высота + широта + сезон)",
        "features": ["h", "latitude_deg",
                     "season_winter", "season_summer", "season_annual"],
        "feature_labels": {
            "h": "Высота, м",
            "latitude_deg": "Широта, °",
            "season": "Сезон (winter/summer/annual)",
        },
        "h_range": (0, 80000),
        "latitude_range": (15, 80),
        "seasons": ["winter", "summer", "annual"],
        "standard": "ISO 5878:2026",
        "description": "Высота, широта и сезон.",
    },
    "4d": {
        "label": "4D — ISO 5878 + влажность",
        "features": ["h", "latitude_deg",
                     "season_winter", "season_summer", "season_annual", "w"],
        "feature_labels": {
            "h": "Высота, м",
            "latitude_deg": "Широта, °",
            "season": "Сезон",
            "w": "Относительная влажность (0..1)",
        },
        "h_range": (0, 10000),
        "latitude_range": (15, 80),
        "seasons": ["winter", "summer", "annual"],
        "w_range": (0, 1),
        "standard": "ISO 5878:2026",
        "description": "Высота, широта, сезон и влажность. Ограничено 10 км "
                       "(влажностные данные ISO 5878 заданы только до 10 км).",
    },
}

TARGETS = ["T", "P", "rho", "a"]
TARGET_LABELS = {
    "T": "Температура, К",
    "P": "Давление, Па",
    "rho": "Плотность, кг/м³",
    "a": "Скорость звука, м/с",
}


# ---------- Загрузка ----------
@lru_cache(maxsize=4)
def load_dataset(name: str) -> pd.DataFrame:
    """
    Загружает датасет по имени (1d/2d/3d/4d) и кэширует.
    Возвращает DataFrame с колонками h, latitude_deg, season, w, T, P, rho, a.
    """
    if name not in DATASET_META:
        raise ValueError(f"Неизвестный датасет: {name}")

    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Датасет не найден: {path}")

    df = pd.read_csv(path)
    return df


def get_dataset_info(name: str) -> dict:
    """Возвращает метаданные датасета + статистику по нему."""
    if name not in DATASET_META:
        raise ValueError(f"Неизвестный датасет: {name}")

    df = load_dataset(name)
    meta = dict(DATASET_META[name])

    # Добавляем статистику
    meta["n_points"] = len(df)
    meta["n_features"] = len(meta["features"])
    meta["n_targets"] = len(TARGETS)
    meta["targets"] = TARGETS
    meta["target_labels"] = TARGET_LABELS

    # Статистика по высоте
    if "h" in df.columns:
        meta["h_min"] = float(df["h"].min())
        meta["h_max"] = float(df["h"].max())

    # Статистика по широте
    if "latitude_deg" in df.columns:
        meta["latitude_min"] = float(df["latitude_deg"].min())
        meta["latitude_max"] = float(df["latitude_deg"].max())

    return meta


def list_datasets() -> list:
    """Возвращает список всех доступных датасетов с краткими метаданными."""
    result = []
    for name in ["1d", "2d", "3d", "4d"]:
        info = get_dataset_info(name)
        result.append({
            "name": name,
            "label": info["label"],
            "n_points": info["n_points"],
            "n_features": info["n_features"],
            "h_range": info.get("h_range"),
            "standard": info["standard"],
            "description": info["description"],
        })
    return result


def get_reference_data(name: str):
    """
    Возвращает эталонные данные для валидации.
    По сути — тот же датасет, но в чистом виде.
    """
    return load_dataset(name)


# ---------- Тест при импорте ----------
if __name__ == "__main__":
    print("=" * 60)
    print("Проверка загрузчика датасетов")
    print("=" * 60)

    for ds in ["1d", "2d", "3d", "4d"]:
        df = load_dataset(ds)
        info = get_dataset_info(ds)
        print(f"\n▶ {ds}: {info['label']}")
        print(f"   Точек: {info['n_points']}")
        print(f"   Признаки: {info['features']}")
        print(f"   Высота: {info.get('h_min', '?')} — {info.get('h_max', '?')} м")
        print(f"   Стандарт: {info['standard']}")