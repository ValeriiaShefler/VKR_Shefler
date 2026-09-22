"""
Универсальный тренер моделей для MSA Studio.

Поддерживает:
    - MLP (многослойный перцептрон) с log-transform и lbfgs
    - GPR (гауссовский процесс) с Matern-ядром

Гиперпараметры по умолчанию — те, что были подобраны в экспериментах.
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..data_loader import load_dataset, get_dataset_info, TARGETS


# ---------- Глобальное хранилище обученных моделей ----------
_MODELS_CACHE = {}   # {cache_key: {"models": {...}, "scalers": {...}, "info": {...}}}


# ---------- Log-transform для P и rho ----------
LOG_PARAMS = {"P": 1, "rho": 2}


def log_transform_y(y):
    y_out = y.astype(float).copy()
    for param, idx in LOG_PARAMS.items():
        y_out[:, idx] = np.log(np.maximum(y[:, idx], 1e-10))
    return y_out


def inverse_log_transform_y(y):
    y_out = y.astype(float).copy()
    for param, idx in LOG_PARAMS.items():
        y_out[:, idx] = np.exp(y_out[:, idx])
    return y_out


# ---------- Гиперпараметры по умолчанию ----------
DEFAULT_HYPERPARAMS = {
    "MLP": {
        "hidden_layer_sizes": (100, 50, 25),
        "activation": "relu",
        "solver": "lbfgs",
        "alpha": 1e-4,
        "max_iter": 5000,
    },
    "GPR": {
        "length_scale": 1.0,
        "length_scale_bounds": (1e-3, 1e3),
        "constant_value": 1.0,
        "constant_value_bounds": (1e-2, 1e3),
        "noise_level": 1e-6,
        "noise_level_bounds": (1e-10, 1e-2),
        "n_restarts_optimizer": 5,
    },
}


def get_default_hyperparams(method: str) -> dict:
    """Возвращает гиперпараметры по умолчанию для метода."""
    if method not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"Неизвестный метод: {method}")
    return dict(DEFAULT_HYPERPARAMS[method])


# ---------- Cache key ----------
def make_cache_key(dataset: str, method: str, hyperparams: dict) -> str:
    """Формирует уникальный ключ для кэша обученных моделей."""
    parts = [dataset, method]
    for k in sorted(hyperparams):
        parts.append(f"{k}={hyperparams[k]}")
    return "|".join(parts)


# ---------- Обучение MLP ----------
def train_mlp(X_train, y_train, hyperparams):
    """Обучает 4 MLP-модели (по одной на T, P, rho, a)."""
    scaler_X = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_train)

    y_tr_t = log_transform_y(y_train)
    scaler_y = StandardScaler()
    y_tr_scaled = scaler_y.fit_transform(y_tr_t)

    models = {}
    t0 = time.perf_counter()
    for i, param in enumerate(TARGETS):
        m = MLPRegressor(
            hidden_layer_sizes=tuple(hyperparams["hidden_layer_sizes"]),
            activation=hyperparams["activation"],
            solver=hyperparams["solver"],
            alpha=hyperparams["alpha"],
            max_iter=hyperparams["max_iter"],
            random_state=42,
        )
        m.fit(X_tr_s, y_tr_scaled[:, i])
        models[param] = m
    train_time = time.perf_counter() - t0

    return {
        "models": models,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_time_s": train_time,
    }


# ---------- Обучение GPR ----------
def train_gpr(X_train, y_train, hyperparams):
    """Обучает 4 GPR-модели (по одной на T, P, rho, a)."""
    scaler_X = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_train)

    y_tr_t = log_transform_y(y_train)
    scaler_y = StandardScaler()
    y_tr_scaled = scaler_y.fit_transform(y_tr_t)

    kernel = (
        ConstantKernel(
            hyperparams["constant_value"],
            constant_value_bounds=hyperparams["constant_value_bounds"],
        )
        * Matern(
            length_scale=hyperparams["length_scale"],
            length_scale_bounds=hyperparams["length_scale_bounds"],
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=hyperparams["noise_level"],
            noise_level_bounds=hyperparams["noise_level_bounds"],
        )
    )

    models = {}
    t0 = time.perf_counter()
    for i, param in enumerate(TARGETS):
        m = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=hyperparams["n_restarts_optimizer"],
            random_state=42,
        )
        m.fit(X_tr_s, y_tr_scaled[:, i])
        models[param] = m
    train_time = time.perf_counter() - t0

    return {
        "models": models,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_time_s": train_time,
    }


# ---------- Обучение (универсальный вход) ----------
def train(dataset: str, method: str, hyperparams: dict = None,
           test_size: float = 0.2) -> dict:
    """
    Обучает модель на датасете.

    Параметры:
        dataset     — "1d" / "2d" / "3d" / "4d"
        method      — "MLP" / "GPR"
        hyperparams — словарь гиперпараметров (None → по умолчанию)
        test_size   — доля теста

    Возвращает:
        dict с полями: cache_key, method, dataset, train_time_s,
                        metrics (MAE/RMSE/R² по параметрам), n_train, n_test
    """
    if hyperparams is None:
        hyperparams = get_default_hyperparams(method)

    cache_key = make_cache_key(dataset, method, hyperparams)
    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]["info"]

    # Загрузка данных
    df = load_dataset(dataset)
    info = get_dataset_info(dataset)

    # One-hot для сезона
    if "season" in df.columns:
        season_dummies = pd.get_dummies(df["season"], prefix="season")
        for col in ["season_winter", "season_summer", "season_annual"]:
            if col not in season_dummies.columns:
                season_dummies[col] = 0
        df = pd.concat([df, season_dummies], axis=1)

    X = df[info["features"]].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Сортировка для 1D (сплайны не нужны — MLP/GPR работают с любым порядком)
    # Но для стабильности оставим в исходном порядке

    # Обучение
    if method == "MLP":
        result = train_mlp(X_train, y_train, hyperparams)
    elif method == "GPR":
        result = train_gpr(X_train, y_train, hyperparams)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # Метрики
    y_pred = _predict_with_result(result, X_test, method)
    metrics = _compute_metrics(y_test, y_pred)

    # Info
    info_out = {
        "cache_key": cache_key,
        "dataset": dataset,
        "method": method,
        "hyperparams": hyperparams,
        "train_time_s": result["train_time_s"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
    }

    # Кэшируем
    _MODELS_CACHE[cache_key] = {
        "models": result,
        "info": info_out,
        "X_test": X_test,
        "y_test": y_test,
    }

    return info_out


# ---------- Предсказание ----------
def _predict_with_result(result: dict, X: np.ndarray, method: str) -> np.ndarray:
    """Предсказание на основе обученной модели."""
    X_s = result["scaler_X"].transform(X)

    preds = np.column_stack([
        result["models"][param].predict(X_s) for param in TARGETS
    ])

    preds_t = result["scaler_y"].inverse_transform(preds)
    preds = inverse_log_transform_y(preds_t)
    return preds


def predict_point(cache_key: str, X: np.ndarray) -> dict:
    """
    Предсказание для одной или нескольких точек.

    X — массив (N, n_features).
    Возвращает dict {T, P, rho, a}.
    """
    if cache_key not in _MODELS_CACHE:
        raise ValueError("Модель не обучена. Сначала вызовите /ai/train.")

    result = _MODELS_CACHE[cache_key]["models"]
    y_pred = _predict_with_result(result, X, method=None)

    # Усредняем по N (для одной точки N=1)
    if len(y_pred) == 1:
        T, P, rho, a = y_pred[0]
    else:
        T, P, rho, a = y_pred.mean(axis=0)

    return {
        "T": float(T),
        "P": float(P),
        "rho": float(rho),
        "a": float(a),
    }


# ---------- Метрики ----------
def _compute_metrics(y_true, y_pred):
    out = {}
    for i, param in enumerate(TARGETS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"MAE_{param}"] = float(mean_absolute_error(yt, yp))
        out[f"RMSE_{param}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        out[f"R2_{param}"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    out["MAE_avg"] = float(np.mean([out[f"MAE_{p}"] for p in TARGETS]))
    out["RMSE_avg"] = float(np.mean([out[f"RMSE_{p}"] for p in TARGETS]))
    out["R2_avg"] = float(np.mean([out[f"R2_{p}"] for p in TARGETS]))
    return out


# ---------- Управление кэшем ----------
def clear_cache():
    """Очищает все обученные модели."""
    _MODELS_CACHE.clear()


def list_trained():
    """Возвращает список всех обученных моделей."""
    return [
        {
            "cache_key": k,
            "dataset": v["info"]["dataset"],
            "method": v["info"]["method"],
            "train_time_s": v["info"]["train_time_s"],
            "metrics_avg": v["info"]["metrics"].get("R2_avg"),
        }
        for k, v in _MODELS_CACHE.items()
    ]


# ---------- Тест при импорте ----------
if __name__ == "__main__":
    print("=" * 60)
    print("Проверка тренера")
    print("=" * 60)

    for method in ["MLP", "GPR"]:
        print(f"\n▶ Обучение {method} на 1D...")
        info = train("1d", method)
        print(f"   Время: {info['train_time_s']:.2f} с")
        print(f"   MAE_avg: {info['metrics']['MAE_avg']:.4f}")
        print(f"   R2_avg:  {info['metrics']['R2_avg']:.4f}")

    print("\n▶ Кэш:")
    for t in list_trained():
        print(f"   {t['cache_key']}: R²={t['metrics_avg']:.4f}")