"""
Единый эксперимент: сравнение методов аппроксимации МСА при разных шагах сетки.

Запуск:
    pip install numpy pandas scipy scikit-learn
    pip install torch          # опционально, для PINN
    python experiments/run_all_experiments.py

Результат:
    results/raw_results.csv               — длинная таблица всех измерений
    results/tables/<metric>.csv           — pivot-таблицы (метод × шаг)
"""
import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Путь до эталонной модели
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.models.isa_core import standard_atmosphere

# Опционально: PyTorch для PINN
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠ PyTorch не найден — PINN будет пропущен")
    print("  Установите: pip install torch")


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
STEPS = [10, 25, 50, 100, 200, 500, 1000]
TEST_STEP = 5                       # мелкая тестовая сетка
PARAMS = ["T", "P", "rho", "a"]
PARAM_NAMES = {"T": "T, K", "P": "P, Pa", "rho": "ρ, kg/m³", "a": "a, m/s"}

# Ограничения на тяжёлые методы
GP_MAX_POINTS = 500                 # GPR — O(n³)
SVR_MAX_POINTS = 500                # SVR медленно на больших выборках
PINN_EPOCHS = 2000                  # эпох обучения PINN


# ============================================================
# ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
def build_isa_table(heights):
    """Возвращает массив [h, T, P, rho, a] для заданных высот."""
    rows = []
    for h in heights:
        T, P, rho, a = standard_atmosphere(float(h))
        rows.append([h, T, P, rho, a])
    return np.array(rows)


ALL_H = np.arange(0, 20001, TEST_STEP)


def make_splits(step):
    """Train с шагом `step`, Test — с шагом TEST_STEP, без пересечения."""
    h_train = np.arange(0, 20001, step)
    h_test = np.setdiff1d(ALL_H, h_train)
    return build_isa_table(h_train), build_isa_table(h_test), h_train, h_test


# ============================================================
# PINN (PyTorch)
# ============================================================
if HAS_TORCH:
    class PINN_ISA(nn.Module):
        """PINN с сигмоидным ограничением выходов в физические диапазоны."""
        def __init__(self, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 4),
            )

        def forward(self, h):
            h_norm = h / 20000.0
            out = self.net(h_norm)
            T = 200.0 + 100.0 * torch.sigmoid(out[:, 0:1])       # [200, 300]
            P = 5000.0 + 97000.0 * torch.sigmoid(out[:, 1:2])   # [5000, 102000]
            rho = 0.08 + 1.22 * torch.sigmoid(out[:, 2:3])      # [0.08, 1.3]
            a = 290.0 + 55.0 * torch.sigmoid(out[:, 3:4])       # [290, 345]
            return T, P, rho, a

    def physics_loss_pinn(model, h_colloc):
        h = h_colloc.clone().requires_grad_(True)
        T, P, rho, a = model(h)
        dP_dh = torch.autograd.grad(P.sum(), h, create_graph=True)[0]
        g, R, gamma = 9.80665, 287.053, 1.4
        L1 = torch.mean((dP_dh + rho * g) ** 2)
        L2 = torch.mean((P - rho * R * T) ** 2)
        L3 = torch.mean((a ** 2 - gamma * R * T) ** 2)
        return L1 + L2 + L3

    def train_pinn(h_train, y_train, epochs=PINN_EPOCHS, lr=1e-3):
        model = PINN_ISA(hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        h_t = torch.tensor(h_train, dtype=torch.float32).reshape(-1, 1)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        h_c = torch.tensor(np.linspace(0, 20000, 300).reshape(-1, 1),
                           dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            T, P, rho, a = model(h_t)
            pred = torch.cat([T, P, rho, a], dim=1)
            L_data = torch.mean((pred - y_t) ** 2)
            L_phys = physics_loss_pinn(model, h_c)
            loss = L_data + 0.01 * L_phys
            loss.backward()
            opt.step()
        return model

    def predict_pinn(model, X):
        with torch.no_grad():
            h_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
            T, P, rho, a = model(h_t)
            return torch.cat([T, P, rho, a], dim=1).numpy()


# ============================================================
# ФИЗИЧЕСКАЯ НЕВЯЗКА (для любых моделей)
# ============================================================
def physics_residual(h_grid, y_pred):
    """
    Относительная невязка трёх физических уравнений.
    y_pred: [n, 4] для T, P, rho, a (в этом порядке).
    """
    T, P, rho, a = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    h = h_grid.ravel()

    # dP/dh — численно (центральная разность)
    dP_dh = np.gradient(P, h)
    g, R, gamma = 9.80665, 287.053, 1.4

    # Относительные невязки (нормируем, чтобы были безразмерны)
    L1 = np.mean(np.abs(dP_dh + rho * g)) / np.mean(np.abs(rho * g))
    L2 = np.mean(np.abs(P - rho * R * T)) / np.mean(P)
    L3 = np.mean(np.abs(a ** 2 - gamma * R * T)) / np.mean(a ** 2)
    return L1, L2, L3, L1 + L2 + L3


# ============================================================
# МЕТРИКИ
# ============================================================
def compute_metrics(y_true, y_pred, train_time, model_obj, predict_fn,
                    h_test, n_train_points):
    """Все метрики одной модели на одном шаге."""
    out = {
        "n_train": n_train_points,
        "n_test": len(y_true),
        "train_time_s": train_time,
    }

    # Точность по каждому параметру
    for i, p in enumerate(PARAMS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"MAE_{p}"] = mean_absolute_error(yt, yp)
        out[f"RMSE_{p}"] = np.sqrt(mean_squared_error(yt, yp))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        out[f"R2_{p}"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[f"MaxErr_{p}"] = np.max(np.abs(yt - yp))
        # MAPE: только там, где yt != 0
        mask = np.abs(yt) > 1e-9
        out[f"MAPE_{p}"] = 100 * np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]))
    # Усреднённые
    for m in ["MAE", "RMSE", "R2", "MaxErr", "MAPE"]:
        vals = [out[f"{m}_{p}"] for p in PARAMS]
        out[f"{m}_avg"] = float(np.mean(vals))

    # Время предсказания (1 точка)
    h_one = np.array([[7500.0]])
    t0 = time.perf_counter()
    for _ in range(500):
        predict_fn(h_one)
    out["time_1point_us"] = (time.perf_counter() - t0) / 500 * 1e6

    # Время предсказания (1000 точек)
    h_batch = np.linspace(0, 20000, 1000).reshape(-1, 1)
    t0 = time.perf_counter()
    for _ in range(30):
        predict_fn(h_batch)
    out["time_1000points_ms"] = (time.perf_counter() - t0) / 30 * 1e3

    # Размер модели
    try:
        out["model_size_kb"] = len(pickle.dumps(model_obj)) / 1024
    except Exception:
        out["model_size_kb"] = np.nan

    # Физическая невязка
    try:
        L1, L2, L3, Lsum = physics_residual(h_test, y_pred)
        out["physics_hydro"] = L1
        out["physics_gas"] = L2
        out["physics_sound"] = L3
        out["physics_total"] = Lsum
    except Exception:
        out["physics_hydro"] = out["physics_gas"] = out["physics_sound"] = np.nan
        out["physics_total"] = np.nan

    # Гладкость: норма второй производной
    try:
        d2 = np.gradient(np.gradient(y_pred[:, 0], h_test.ravel()),
                         h_test.ravel())
        out["smoothness_T"] = float(np.mean(np.abs(d2)))
    except Exception:
        out["smoothness_T"] = np.nan

    return out


# ============================================================
# ОБЁРТКИ МЕТОДОВ
# ============================================================
def fit_predict(X_train, y_train, X_test, method, h_train, h_test):
    """Возвращает (y_pred, train_time, model_obj, predict_fn)."""
    h_tr = h_train.ravel()

    # ---- 1. Линейная интерполяция ----
    if method == "Линейная интерполяция":
        f = interp1d(h_tr, y_train, axis=0, kind="linear", fill_value="extrapolate")
        return f(h_test.ravel()), 0.0, f, lambda x: f(x.ravel())

    # ---- 2. Кубический сплайн ----
    if method == "Кубический сплайн":
        f = CubicSpline(h_tr, y_train, axis=0)
        return f(h_test.ravel()), 0.0, f, lambda x: f(x.ravel())

    # ---- 3. PCHIP ----
    if method == "PCHIP":
        f = PchipInterpolator(h_tr, y_train, axis=0)
        return f(h_test.ravel()), 0.0, f, lambda x: f(x.ravel())

    # ---- 4. Полином 15-й степени ----
    if method == "Полином 15-й степени":
        coeffs = np.polyfit(h_tr, y_train, deg=15)
        def predict(x):
            return np.column_stack([np.polyval(coeffs[:, i], x.ravel())
                                    for i in range(y_train.shape[1])])
        return predict(h_test), 0.0, coeffs, predict

    # ---- 5. Линейная регрессия ----
    if method == "Линейная регрессия":
        m = LinearRegression().fit(X_train, y_train)
        return m.predict(X_test), 0.0, m, m.predict

    # ---- 6. Полиномиальная регрессия (степень 5) ----
    if method == "Полиномиальная регрессия (5)":
        poly = PolynomialFeatures(degree=5, include_bias=False)
        Xtr = poly.fit_transform(X_train)
        Xte = poly.transform(X_test)
        m = LinearRegression().fit(Xtr, y_train)
        return m.predict(Xte), 0.0, m, lambda x: m.predict(poly.transform(x))

    # ---- 7. MLP ----
    if method == "MLP (50,25)":
        poly = PolynomialFeatures(degree=3, include_bias=False)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(poly.fit_transform(X_train))
        Xte = scaler.transform(poly.transform(X_test))
        m = MLPRegressor(hidden_layer_sizes=(50, 25), activation="relu",
                         max_iter=3000, random_state=42)
        t0 = time.perf_counter()
        m.fit(Xtr, y_train)
        dt = time.perf_counter() - t0
        return m.predict(Xte), dt, m, lambda x: m.predict(
            scaler.transform(poly.transform(x)))

    # ---- 8. Градиентный бустинг ----
    if method == "Градиентный бустинг":
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                       random_state=42)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        return m.predict(X_test), dt, m, m.predict

    # ---- 9. Случайный лес ----
    if method == "Случайный лес":
        m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   random_state=42, n_jobs=-1)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        return m.predict(X_test), dt, m, m.predict

    # ---- 10. SVR ----
    if method == "SVR (RBF)":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        m = SVR(kernel="rbf", C=100.0, epsilon=0.01, gamma="scale")
        t0 = time.perf_counter()
        m.fit(Xtr, y_train)
        dt = time.perf_counter() - t0
        return m.predict(Xte), dt, m, lambda x: m.predict(
            scaler.transform(x))

    # ---- 11. Гауссовский процесс ----
    if method == "Гауссовский процесс":
        if len(X_train) > GP_MAX_POINTS:
            return None, None, None, None
        m = GaussianProcessRegressor(normalize_y=True, alpha=1e-6,
                                      random_state=42)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        return m.predict(X_test), dt, m, m.predict

    # ---- 12. PINN ----
    if method == "PINN":
        if not HAS_TORCH:
            return None, None, None, None
        t0 = time.perf_counter()
        m = train_pinn(h_train, y_train, epochs=PINN_EPOCHS)
        dt = time.perf_counter() - t0
        return predict_pinn(m, h_test), dt, m, lambda x: predict_pinn(m, x)

    raise ValueError(f"Неизвестный метод: {method}")


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    methods = [
        "Линейная интерполяция",
        "Кубический сплайн",
        "PCHIP",
        "Полином 15-й степени",
        "Линейная регрессия",
        "Полиномиальная регрессия (5)",
        "MLP (50,25)",
        "Градиентный бустинг",
        "Случайный лес",
        "SVR (RBF)",
        "Гауссовский процесс",
    ]
    if HAS_TORCH:
        methods.append("PINN")

    all_rows = []

    for step in STEPS:
        train, test, h_train, h_test = make_splits(step)
        X_train = train[:, 0:1]
        y_train = train[:, 1:]
        X_test = test[:, 0:1]
        y_test = test[:, 1:]

        print(f"\n▶ Шаг = {step} м (train={len(h_train)}, test={len(h_test)})")

        for method in methods:
            try:
                y_pred, t_fit, model_obj, predict_fn = fit_predict(
                    X_train, y_train, X_test, method, h_train, h_test
                )
                if y_pred is None:
                    print(f"   [SKIP] {method}")
                    continue
                metrics = compute_metrics(
                    y_test, y_pred, t_fit, model_obj, predict_fn,
                    h_test, len(h_train)
                )
                metrics["method"] = method
                metrics["step"] = step
                all_rows.append(metrics)
                print(f"   ✓ {method:<30} R²(T)={metrics['R2_T']:.4f}  "
                      f"1pt={metrics['time_1point_us']:.1f}мкс")
            except Exception as e:
                print(f"   ✗ {method}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv("results/raw_results.csv", index=False)
    print(f"\n✅ Длинная таблица: results/raw_results.csv ({len(df)} строк)")

    # Pivot-таблицы
    metric_names = [
        "MAE_T", "MAE_P", "MAE_rho", "MAE_a", "MAE_avg",
        "RMSE_T", "RMSE_P", "RMSE_rho", "RMSE_a", "RMSE_avg",
        "R2_T", "R2_P", "R2_rho", "R2_a", "R2_avg",
        "MAPE_T", "MAPE_P", "MAPE_rho", "MAPE_a",
        "MaxErr_T", "MaxErr_P", "MaxErr_rho", "MaxErr_a",
        "time_1point_us", "time_1000points_ms",
        "model_size_kb", "train_time_s",
        "physics_hydro", "physics_gas", "physics_sound", "physics_total",
        "smoothness_T",
    ]

    print("\n=== Создание pivot-таблиц ===")
    for metric in metric_names:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index="method", columns="step",
                                values=metric, aggfunc="mean")
        pivot = pivot.reindex([m for m in methods if m in pivot.index])
        pivot.to_csv(f"results/tables/{metric}.csv")
        print(f"  ✓ {metric}.csv")

    # Красивый вывод ключевых таблиц на экран
    print("\n" + "=" * 80)
    print("MAE для температуры (K)")
    print("=" * 80)
    p = df.pivot_table(index="method", columns="step",
                       values="MAE_T", aggfunc="mean")
    print(p.round(3).to_string())

    print("\n" + "=" * 80)
    print("R² для скорости звука")
    print("=" * 80)
    p = df.pivot_table(index="method", columns="step",
                       values="R2_a", aggfunc="mean")
    print(p.round(4).to_string())

    print("\n" + "=" * 80)
    print("Физическая невязка (суммарная)")
    print("=" * 80)
    p = df.pivot_table(index="method", columns="step",
                       values="physics_total", aggfunc="mean")
    print(p.round(4).to_string())

    print("\n✅ Готово. Все таблицы в results/tables/")