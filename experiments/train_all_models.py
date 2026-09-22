"""
Обучение всех моделей на 4 датасетах (1D, 2D, 3D, 4D).
Сравнение интерполяции и ML-методов по точности, скорости и памяти.

Запуск:
    python experiments/train_all_models.py

Результат:
    results/training/raw_results.csv
    results/training/tables/<metric>.csv
    results/training/summary.csv
"""
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, RBFInterpolator

warnings.filterwarnings("ignore")

# ---------- Пути ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "training")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------- Параметры ----------
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGETS = ["T", "P", "rho", "a"]

METHOD_GROUPS = {
    "Классическая интерполяция": [
        "Линейная интерполяция",
        "Кубический сплайн",
        "PCHIP",
        "RBF-интерполяция",
    ],
    "ML-регрессия": [
        "MLP",
        "GPR",
        "GradientBoosting",
        "RandomForest",
        "SVR",
    ],
    "Baseline": [
        "LinearRegression",
        "PolynomialFeatures(5)+LR",
    ],
}

# Плоский порядок для сортировки
METHOD_ORDER = []
for group in METHOD_GROUPS.values():
    METHOD_ORDER.extend(group)


def sort_methods(df):
    """Сортирует строки DataFrame по группам методов."""
    df = df.copy()
    df["_group_order"] = df.index.map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999
    )
    df = df.sort_values("_group_order").drop(columns="_group_order")
    return df

GP_MAX_POINTS = 600
SVR_MAX_POINTS = 1000

# Индексы параметров для log-transform
LOG_PARAMS = {"P": 1, "rho": 2}  # T и a остаются как есть

# ============================================================
# 0. Log-трансформация выходов
# ============================================================
def log_transform_y(y):
    """Преобразует P и rho в логарифм, T и a оставляет как есть."""
    y_out = y.astype(float).copy()
    for param, idx in LOG_PARAMS.items():
        # Защита от нулей и отрицательных значений
        y_out[:, idx] = np.log(np.maximum(y[:, idx], 1e-10))
    return y_out


def inverse_log_transform_y(y):
    """Обратное преобразование: exp для P и rho."""
    y_out = y.astype(float).copy()
    for param, idx in LOG_PARAMS.items():
        y_out[:, idx] = np.exp(y_out[:, idx])
    return y_out


# ============================================================
# 1. Загрузка датасетов
# ============================================================
def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    return pd.read_csv(path)


def prepare_features(df, name):
    if "season" in df.columns:
        season_dummies = pd.get_dummies(df["season"], prefix="season")
        for col in ["season_winter", "season_summer", "season_annual"]:
            if col not in season_dummies.columns:
                season_dummies[col] = 0
        df = pd.concat([df, season_dummies], axis=1)

    if name == "1d":
        feature_cols = ["h"]
    elif name == "2d":
        feature_cols = ["h", "latitude_deg"]
    elif name == "3d":
        feature_cols = ["h", "latitude_deg",
                        "season_winter", "season_summer", "season_annual"]
    elif name == "4d":
        feature_cols = ["h", "latitude_deg",
                        "season_winter", "season_summer", "season_annual", "w"]
    else:
        raise ValueError(f"Неизвестный датасет: {name}")

    return df[feature_cols].values, df[TARGETS].values, feature_cols


# ============================================================
# 2. Метрики
# ============================================================
def compute_metrics(y_true, y_pred, train_time, model_obj, predict_fn,
                    X_test, X_train, y_train):
    out = {"train_time_s": train_time}

    for i, param in enumerate(TARGETS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"MAE_{param}"] = mean_absolute_error(yt, yp)
        out[f"RMSE_{param}"] = np.sqrt(mean_squared_error(yt, yp))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        out[f"R2_{param}"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    for m in ["MAE", "RMSE", "R2"]:
        out[f"{m}_avg"] = float(np.mean([out[f"{m}_{p}"] for p in TARGETS]))

    # Overfit ratio
    try:
        y_train_pred = predict_fn(X_train)
        train_mae = np.mean([
            mean_absolute_error(y_train[:, i], y_train_pred[:, i])
            for i in range(4)
        ])
        out["overfit_ratio"] = out["MAE_avg"] / train_mae if train_mae > 0 else np.nan
    except Exception:
        out["overfit_ratio"] = np.nan

    # Время 1 точки
    x_one = X_test[0:1]
    t0 = time.perf_counter()
    for _ in range(100):
        predict_fn(x_one)
    out["time_1point_us"] = (time.perf_counter() - t0) / 100 * 1e6

    # Время 1000 точек
    if len(X_test) >= 1000:
        x_batch = X_test[:1000]
    else:
        x_batch = np.tile(X_test, (1000 // len(X_test) + 1, 1))[:1000]
    t0 = time.perf_counter()
    for _ in range(10):
        predict_fn(x_batch)
    out["time_1000points_ms"] = (time.perf_counter() - t0) / 10 * 1e3

    # Размер модели
    try:
        out["model_size_kb"] = len(pickle.dumps(model_obj)) / 1024
    except Exception:
        out["model_size_kb"] = np.nan

    return out


# ============================================================
# 3. Обёртки
# ============================================================
def fit_1d_interpolation(X_train, y_train, X_test, method):
    x_tr = X_train.ravel()
    x_te = X_test.ravel()

    if method == "Линейная интерполяция":
        f = interp1d(x_tr, y_train, axis=0, kind="linear", fill_value="extrapolate")
        return f(x_te), 0.0, f, lambda x: f(x.ravel())

    if method == "Кубический сплайн":
        f = CubicSpline(x_tr, y_train, axis=0, extrapolate=True)
        return f(x_te), 0.0, f, lambda x: f(x.ravel())

    if method == "PCHIP":
        f = PchipInterpolator(x_tr, y_train, axis=0, extrapolate=True)
        return f(x_te), 0.0, f, lambda x: f(x.ravel())

    raise ValueError(method)


def fit_rbf_interpolation(X_train, y_train, X_test, method):
    if len(X_train) > 500:
        idx = np.random.RandomState(42).choice(len(X_train), 500, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    t0 = time.perf_counter()
    rbf = RBFInterpolator(X_train, y_train, kernel="thin_plate_spline",
                           smoothing=0.0)
    dt = time.perf_counter() - t0
    y_pred = rbf(X_test)
    return y_pred, dt, rbf, rbf


def fit_ml_model(X_train, y_train, X_test, method):
    """
    Обучение ML-метода.
    PolynomialFeatures обрабатывается отдельно (не через scaler).
    Для MLP и GPR — log-transform для P и rho + стандартизация выходов.
    """
    # ---------- ОТДЕЛЬНАЯ ВЕТКА: PolynomialFeatures(5)+LR ----------
    if method == "PolynomialFeatures(5)+LR":
        preds_all = []
        models = {}
        t_total = 0.0
        for i, param in enumerate(TARGETS):
            poly = PolynomialFeatures(degree=5, include_bias=False)
            X_tr_p = poly.fit_transform(X_train)
            X_te_p = poly.transform(X_test)
            m = LinearRegression()
            t0 = time.perf_counter()
            m.fit(X_tr_p, y_train[:, i])
            t_total += time.perf_counter() - t0
            preds_all.append(m.predict(X_te_p))
            models[param] = (m, poly)
        y_pred = np.column_stack(preds_all)

        def predict_fn(X):
            outputs = []
            for param in TARGETS:
                m, poly = models[param]
                outputs.append(m.predict(poly.transform(X)))
            return np.column_stack(outputs)

        return y_pred, t_total, models, predict_fn

    # ---------- ВСЕ ОСТАЛЬНЫЕ МЕТОДЫ ----------
    need_scaling = method in ["MLP", "MLP-v2", "GPR", "SVR"]
    need_log = method in ["MLP", "MLP-v2", "GPR"]

    # Стандартизация входов
    if need_scaling:
        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_train)
        X_te_s = scaler_X.transform(X_test)
    else:
        scaler_X = None
        X_tr_s = X_train
        X_te_s = X_test

    # Log-transform и стандартизация выходов
    if need_log:
        y_train_t = log_transform_y(y_train)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_t)
    else:
        scaler_y = None
        y_train_scaled = y_train

    # ... дальше идёт существующий код с build_model() и циклом

    def build_model():
        if method == "LinearRegression":
            return LinearRegression()
        if method == "MLP":
            return MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="lbfgs",
                alpha=1e-4,              # ← L2-регуляризация (было по умолчанию 1e-4, но явно)
                max_iter=5000,
                random_state=RANDOM_STATE,
            )

        if method == "GPR":
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e3))
                * Matern(
                    length_scale=1.0,
                    length_scale_bounds=(1e-3, 1e3),    # ← расширено
                    nu=2.5,
                )
                + WhiteKernel(
                    noise_level=1e-6,
                    noise_level_bounds=(1e-10, 1e-2),   # ← расширено
                )
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=5,                 # ← было 1
                random_state=RANDOM_STATE,
            )
        if method == "GradientBoosting":
            return GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                            random_state=RANDOM_STATE)
        if method == "RandomForest":
            return RandomForestRegressor(n_estimators=100, max_depth=10,
                                        random_state=RANDOM_STATE, n_jobs=-1)
        if method == "SVR":
            return SVR(kernel="rbf", C=100.0, epsilon=0.01)
        raise ValueError(method)

    models = {}
    t_total = 0.0
    preds_all = []

    for i, param in enumerate(TARGETS):
        m = build_model()

        if method == "PolynomialFeatures(5)+LR":
            # Отдельная обработка
            poly = PolynomialFeatures(degree=5, include_bias=False)
            X_tr_p = poly.fit_transform(X_train)
            X_te_p = poly.transform(X_test)
            m = LinearRegression()
            t0 = time.perf_counter()
            m.fit(X_tr_p, y_train[:, i])
            t_total += time.perf_counter() - t0
            preds_all.append(m.predict(X_te_p))
            models[param] = (m, poly)
            continue

        t0 = time.perf_counter()
        if need_log:
            m.fit(X_tr_s, y_train_scaled[:, i])
        else:
            m.fit(X_tr_s, y_train[:, i])
        t_total += time.perf_counter() - t0

        pred = m.predict(X_te_s)
        preds_all.append(pred)
        models[param] = m

    # Собираем предсказания в матрицу
    y_pred_scaled = np.column_stack(preds_all)

    # Обратное преобразование
    if need_log:
        y_pred_t = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = inverse_log_transform_y(y_pred_t)
    else:
        y_pred = y_pred_scaled

    # Универсальная функция предсказания
    def predict_fn(X):
        if method == "PolynomialFeatures(5)+LR":
            outputs = []
            for param in TARGETS:
                m, poly = models[param]
                outputs.append(m.predict(poly.transform(X)))
            return np.column_stack(outputs)

        X_s = scaler_X.transform(X) if scaler_X is not None else X
        outputs = np.column_stack([models[param].predict(X_s)
                                    for param in TARGETS])
        if need_log:
            outputs_t = scaler_y.inverse_transform(outputs)
            outputs = inverse_log_transform_y(outputs_t)
        return outputs

    return y_pred, t_total, models, predict_fn


# ============================================================
# 4. Основной цикл
# ============================================================
METHODS_1D = [
    "Линейная интерполяция",
    "Кубический сплайн",
    "PCHIP",
    "LinearRegression",
    "PolynomialFeatures(5)+LR",
    "MLP",              # ← добавлено
    "GPR",
    "GradientBoosting",
    "RandomForest",
    "SVR",
]

METHODS_ND = [
    "RBF-интерполяция",
    "LinearRegression",
    "MLP",             # ← добавлено
    "GPR",
    "GradientBoosting",
    "RandomForest",
    "SVR",
]


def run_for_dataset(name):
    print(f"\n{'='*60}")
    print(f"ДАТАСЕТ: {name.upper()}")
    print(f"{'='*60}")

    df = load_dataset(name)
    X, y, feature_cols = prepare_features(df, name)
    print(f"X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # КРИТИЧНО: сортировка для сплайнов (только 1D)
    if name == "1d":
        sort_idx_train = np.argsort(X_train[:, 0])
        sort_idx_test = np.argsort(X_test[:, 0])
        X_train = X_train[sort_idx_train]
        y_train = y_train[sort_idx_train]
        X_test = X_test[sort_idx_test]
        y_test = y_test[sort_idx_test]

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    methods = METHODS_1D if name == "1d" else METHODS_ND
    results = []

    for method in methods:
        try:
            print(f"\n  ▶ {method}...", end=" ", flush=True)

            if name == "1d":
                if method in ["Линейная интерполяция", "Кубический сплайн", "PCHIP"]:
                    y_pred, dt, model, pred_fn = fit_1d_interpolation(
                        X_train, y_train, X_test, method
                    )
                else:
                    y_pred, dt, model, pred_fn = fit_ml_model(
                        X_train, y_train, X_test, method
                    )
            else:
                if method == "RBF-интерполяция":
                    y_pred, dt, model, pred_fn = fit_rbf_interpolation(
                        X_train, y_train, X_test, method
                    )
                else:
                    if method == "GPR" and len(X_train) > GP_MAX_POINTS:
                        print(f"[SKIP: n={len(X_train)} > {GP_MAX_POINTS}]")
                        continue
                    if method == "SVR" and len(X_train) > SVR_MAX_POINTS:
                        print(f"[SKIP: n={len(X_train)} > {SVR_MAX_POINTS}]")
                        continue
                    y_pred, dt, model, pred_fn = fit_ml_model(
                        X_train, y_train, X_test, method
                    )

            metrics = compute_metrics(
                y_test, y_pred, dt, model, pred_fn,
                X_test, X_train, y_train
            )
            metrics["method"] = method
            metrics["dataset"] = name
            results.append(metrics)

            print(f"MAE_avg={metrics['MAE_avg']:.4f}  "
                  f"R²_avg={metrics['R2_avg']:.4f}  "
                  f"1pt={metrics['time_1point_us']:.1f}мкс  "
                  f"size={metrics['model_size_kb']:.1f}КБ")
        except Exception as e:
            print(f"❌ {e}")

    return results


# ============================================================
# 5. Главный пайплайн
# ============================================================
if __name__ == "__main__":
    all_results = []
    for dataset in ["1d", "2d", "3d", "4d"]:
        results = run_for_dataset(dataset)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)
    print(f"\n✅ Длинная таблица: {OUT_DIR}/raw_results.csv "
          f"({len(df)} строк)")

    metric_names = [c for c in df.columns if c not in ["method", "dataset"]]

    for metric in metric_names:
        pivot = df.pivot_table(index="method", columns="dataset",
                                values=metric, aggfunc="mean")
        pivot = sort_methods(pivot)
        pivot.to_csv(os.path.join(TABLES_DIR, f"{metric}.csv"))
    print(f"✅ Pivot-таблицы: {TABLES_DIR}/ ({len(metric_names)} файлов)")

    summary = df.pivot_table(
        index="method",
        columns="dataset",
        values=["MAE_avg", "R2_avg", "time_1point_us",
                "model_size_kb", "overfit_ratio"],
        aggfunc="mean"
    )
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    print(f"✅ Сводка: {OUT_DIR}/summary.csv")

    print("\n" + "=" * 70)
    print("MAE_avg")
    print("=" * 70)
    print(sort_methods(df.pivot_table(index="method", columns="dataset",
                          values="MAE_avg")).round(4).to_string())

    print("\n" + "=" * 70)
    print("R²_avg")
    print("=" * 70)
    print(sort_methods(df.pivot_table(index="method", columns="dataset",
                          values="R2_avg")).round(4).to_string())

    print("\n" + "=" * 70)
    print("Размер модели, КБ")
    print("=" * 70)
    print(sort_methods(df.pivot_table(index="method", columns="dataset",
                          values="model_size_kb")).round(2).to_string())

    print("\n" + "=" * 70)
    print("Время предсказания 1 точки, мкс")
    print("=" * 70)
    print(sort_methods(df.pivot_table(index="method", columns="dataset",
                          values="time_1point_us")).round(2).to_string())

    print("\n" + "=" * 70)
    print("Overfit ratio")
    print("=" * 70)
    print(sort_methods(df.pivot_table(index="method", columns="dataset",
                          values="overfit_ratio")).round(3).to_string())