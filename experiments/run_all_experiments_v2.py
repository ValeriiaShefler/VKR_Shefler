"""
experiments/run_all_experiments_v2.py

Сравнение методов аппроксимации МСА:
  - интерполяция (линейная, кубический сплайн, PCHIP, полином)
  - ML (линейная регрессия, полиномиальная, MLP, GB, RF, SVR, GPR, PINN)
  - 3 режима: интерполяция, экстраполяция, зашумлённые данные

Запуск:
    python experiments/run_all_experiments_v2.py

Результат:
    <project_root>/results_v2/raw_results.csv
    <project_root>/results_v2/tables/<metric>__<mode>.csv
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------- Пути (абсолютные) ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v2")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
from app.models.isa_core import standard_atmosphere

# ---------- PyTorch ----------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠ PyTorch не найден — PINN будет пропущен")


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
STEPS = [10, 25, 50, 100, 200, 500, 1000]
PARAMS = ["T", "P", "rho", "a"]
MODES = ["interpolation", "extrapolation", "noisy"]
NOISE_LEVEL = 0.005           # 0.5% шума
EXTRAP_TRAIN_MAX = 15000      # обучаем на 0..15 км
TEST_STEP = 5
GP_MAX_POINTS = 500
PINN_EPOCHS = 3000
PINN_PHYS_WEIGHT = 0.1        # было 0.01 — увеличили в 10 раз


# ============================================================
# ДАННЫЕ
# ============================================================
def build_isa_table(heights):
    rows = []
    for h in heights:
        T, P, rho, a = standard_atmosphere(float(h))
        rows.append([h, T, P, rho, a])
    return np.array(rows)


def make_splits(step, mode):
    """Возвращает train/test массивы [h, T, P, rho, a]."""
    if mode == "interpolation":
        h_train = np.arange(0, 20001, step)
        h_all = np.arange(0, 20001, TEST_STEP)
        h_test = np.setdiff1d(h_all, h_train)

    elif mode == "extrapolation":
        # train: 0..15 км, test: 15..20 км
        h_train = np.arange(0, EXTRAP_TRAIN_MAX + 1, step)
        h_test = np.arange(EXTRAP_TRAIN_MAX + TEST_STEP, 20001, TEST_STEP)

    elif mode == "noisy":
        h_train = np.arange(0, 20001, step)
        h_all = np.arange(0, 20001, TEST_STEP)
        h_test = np.setdiff1d(h_all, h_train)

    train = build_isa_table(h_train)
    test = build_isa_table(h_test)

    if mode == "noisy":
        # Зашумляем только целевые параметры (не высоту)
        rng = np.random.default_rng(42)
        scale = np.std(train[:, 1:], axis=0) * NOISE_LEVEL
        train[:, 1:] = train[:, 1:] + rng.normal(0, scale, size=train[:, 1:].shape)

    return train, test, h_train, h_test


# ============================================================
# PINN (per-parameter)
# ============================================================
if HAS_TORCH:
    class PINN_ISA(nn.Module):
        """PINN для одного параметра с физическими ограничениями."""
        def __init__(self, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, h):
            h_norm = h / 20000.0
            return self.net(h_norm)

    def train_pinn_per_param(h_train, y_train, param_idx, epochs=PINN_EPOCHS):
        """
        Обучает одну PINN для одного параметра.
        param_idx: 0=T, 1=P, 2=rho, 3=a
        """
        model = PINN_ISA(hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        h_t = torch.tensor(h_train, dtype=torch.float32).reshape(-1, 1)
        y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        h_c = torch.tensor(np.linspace(0, 20000, 300).reshape(-1, 1),
                           dtype=torch.float32)

        g, R, gamma = 9.80665, 287.053, 1.4

        for epoch in range(epochs):
            opt.zero_grad()
            pred = model(h_t)
            L_data = torch.mean((pred - y_t) ** 2)

            # Физические ограничения только для соответствующих параметров
            h_c_req = h_c.clone().requires_grad_(True)
            pred_c = model(h_c_req)
            L_phys = torch.tensor(0.0)

            if param_idx == 1:  # P — гидростатика (используем эталонные rho)
                dP_dh = torch.autograd.grad(pred_c.sum(), h_c_req, create_graph=True)[0]
                rho_ref = torch.tensor(
                    [standard_atmosphere(float(h))[2] for h in h_c.numpy().ravel()],
                    dtype=torch.float32).reshape(-1, 1)
                L_phys = torch.mean((dP_dh + rho_ref * g) ** 2) / (1e5 ** 2)

            elif param_idx == 3:  # a — скорость звука (a² = γRT, T из эталона)
                T_ref = torch.tensor(
                    [standard_atmosphere(float(h))[0] for h in h_c.numpy().ravel()],
                    dtype=torch.float32).reshape(-1, 1)
                L_phys = torch.mean((pred_c ** 2 - gamma * R * T_ref) ** 2) / (3e2 ** 4)

            loss = L_data + PINN_PHYS_WEIGHT * L_phys
            loss.backward()
            opt.step()

        return model

    def predict_pinn(model, X):
        with torch.no_grad():
            h_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
            return model(h_t).numpy().ravel()


# ============================================================
# ФИЗИЧЕСКАЯ НЕВЯЗКА (для итогового набора)
# ============================================================
def physics_residual(h_grid, y_pred):
    T, P, rho, a = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    h = h_grid.ravel()
    dP_dh = np.gradient(P, h)
    g, R, gamma = 9.80665, 287.053, 1.4
    L1 = np.mean(np.abs(dP_dh + rho * g)) / np.mean(np.abs(rho * g))
    L2 = np.mean(np.abs(P - rho * R * T)) / np.mean(P)
    L3 = np.mean(np.abs(a ** 2 - gamma * R * T)) / np.mean(a ** 2)
    return L1, L2, L3, L1 + L2 + L3


# ============================================================
# МЕТРИКИ
# ============================================================
def compute_metrics(y_true, y_pred, train_time, model_obj, predict_fn,
                    h_test, n_train, param_indices=range(4)):
    out = {"n_train": n_train, "n_test": len(y_true), "train_time_s": train_time}

    for i, p in enumerate(PARAMS):
        if i not in param_indices:
            continue
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"MAE_{p}"] = mean_absolute_error(yt, yp)
        out[f"RMSE_{p}"] = np.sqrt(mean_squared_error(yt, yp))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        out[f"R2_{p}"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[f"MaxErr_{p}"] = np.max(np.abs(yt - yp))
        mask = np.abs(yt) > 1e-9
        out[f"MAPE_{p}"] = 100 * np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]))

    # Усреднённые (только по доступным параметрам)
    avail = [p for i, p in enumerate(PARAMS) if i in param_indices]
    for m in ["MAE", "RMSE", "R2"]:
        out[f"{m}_avg"] = float(np.mean([out[f"{m}_{p}"] for p in avail]))

    # Время (1 точка)
    h_one = np.array([[7500.0]])
    t0 = time.perf_counter()
    for _ in range(500):
        predict_fn(h_one)
    out["time_1point_us"] = (time.perf_counter() - t0) / 500 * 1e6

    # Размер
    try:
        out["model_size_kb"] = len(pickle.dumps(model_obj)) / 1024
    except Exception:
        out["model_size_kb"] = np.nan

    # Физика — только если есть все 4 параметра
    if len(avail) == 4:
        try:
            L1, L2, L3, Lsum = physics_residual(h_test, y_pred)
            out["physics_hydro"] = L1
            out["physics_gas"] = L2
            out["physics_sound"] = L3
            out["physics_total"] = Lsum
        except Exception:
            out["physics_total"] = np.nan

    return out


# ============================================================
# ОБЁРТКИ МЕТОДОВ (по одному параметру)
# ============================================================
def fit_method_per_param(h_train, y_train_param, h_test, method):
    """Обучает один метод для одного параметра. Возвращает (pred, t, model, fn)."""
    h_tr = h_train.ravel()
    X_tr = h_tr.reshape(-1, 1)
    X_te = h_test.reshape(-1, 1)

    # ---- Интерполяция ----
    if method == "Линейная интерполяция":
        f = interp1d(h_tr, y_train_param, kind="linear", fill_value="extrapolate")
        return f(h_test), 0.0, f, lambda x: f(x.ravel())

    if method == "Кубический сплайн":
        f = CubicSpline(h_tr, y_train_param, extrapolate=True)
        return f(h_test), 0.0, f, lambda x: f(x.ravel())

    if method == "PCHIP":
        f = PchipInterpolator(h_tr, y_train_param, extrapolate=True)
        return f(h_test), 0.0, f, lambda x: f(x.ravel())

    if method == "Полином 15-й степени":
        c = np.polyfit(h_tr, y_train_param, deg=15)
        return np.polyval(c, h_test), 0.0, c, lambda x: np.polyval(c, x.ravel())

    # ---- ML ----
    if method == "Линейная регрессия":
        m = LinearRegression().fit(X_tr, y_train_param)
        return m.predict(X_te), 0.0, m, m.predict

    if method == "Полиномиальная регрессия (5)":
        poly = PolynomialFeatures(degree=5, include_bias=False)
        m = LinearRegression().fit(poly.fit_transform(X_tr), y_train_param)
        return (m.predict(poly.transform(X_te)), 0.0, m,
                lambda x: m.predict(poly.transform(x)))

    if method == "MLP (50,25)":
        poly = PolynomialFeatures(degree=3, include_bias=False)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(poly.fit_transform(X_tr))
        Xte = scaler.transform(poly.transform(X_te))
        m = MLPRegressor(hidden_layer_sizes=(50, 25), activation="relu",
                         max_iter=3000, random_state=42)
        t0 = time.perf_counter(); m.fit(Xtr, y_train_param); dt = time.perf_counter() - t0
        return (m.predict(Xte), dt, m,
                lambda x: m.predict(scaler.transform(poly.transform(x))))

    if method == "Градиентный бустинг":
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        t0 = time.perf_counter(); m.fit(X_tr, y_train_param); dt = time.perf_counter() - t0
        return m.predict(X_te), dt, m, m.predict

    if method == "Случайный лес":
        m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   random_state=42, n_jobs=-1)
        t0 = time.perf_counter(); m.fit(X_tr, y_train_param); dt = time.perf_counter() - t0
        return m.predict(X_te), dt, m, m.predict

    if method == "SVR (RBF)":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tr); Xte = scaler.transform(X_te)
        m = SVR(kernel="rbf", C=100.0, epsilon=0.01, gamma="scale")
        t0 = time.perf_counter(); m.fit(Xtr, y_train_param); dt = time.perf_counter() - t0
        return (m.predict(Xte), dt, m,
                lambda x: m.predict(scaler.transform(x)))

    if method == "Гауссовский процесс":
        if len(X_tr) > GP_MAX_POINTS:
            return None, None, None, None
        kernel = ConstantKernel(1.0) * Matern(length_scale=1e3, nu=2.5) + WhiteKernel(1e-6)
        m = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=42)
        t0 = time.perf_counter(); m.fit(X_tr, y_train_param); dt = time.perf_counter() - t0
        return m.predict(X_te), dt, m, m.predict

    if method == "PINN":
        if not HAS_TORCH:
            return None, None, None, None
        # param_idx определяется по имени — передадим через замыкание
        raise RuntimeError("PINN обрабатывается отдельно")

    raise ValueError(f"Неизвестный метод: {method}")


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
if __name__ == "__main__":
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

    for mode in MODES:
        print(f"\n{'='*70}\nРЕЖИМ: {mode}\n{'='*70}")
        for step in STEPS:
            train, test, h_train, h_test = make_splits(step, mode)
            y_train_full = train[:, 1:]
            y_test_full = test[:, 1:]
            print(f"\n▶ Шаг = {step} м (train={len(h_train)}, test={len(h_test)})")

            for method in methods:
                try:
                    # Для каждого параметра обучаем отдельно
                    y_pred_full = np.zeros_like(y_test_full)
                    t_fit_total = 0.0
                    models_for_size = {}
                    predict_fns = []

                    for i, p in enumerate(PARAMS):
                        if method == "PINN":
                            if not HAS_TORCH:
                                y_pred_full = None; break
                            t0 = time.perf_counter()
                            m = train_pinn_per_param(h_train, y_train_full[:, i], param_idx=i)
                            t_fit_total += time.perf_counter() - t0
                            y_pred_full[:, i] = predict_pinn(m, h_test)
                            models_for_size[p] = m
                            predict_fns.append(lambda x, mm=m: predict_pinn(mm, x))
                        else:
                            pred, dt, m, fn = fit_method_per_param(
                                h_train, y_train_full[:, i], h_test, method)
                            if pred is None:
                                y_pred_full = None; break
                            t_fit_total += dt
                            y_pred_full[:, i] = pred
                            models_for_size[p] = m
                            predict_fns.append(fn)

                    if y_pred_full is None:
                        print(f"   [SKIP] {method}")
                        continue

                    # Для метрик времени используем первую функцию (все одного типа)
                    metrics = compute_metrics(
                        y_test_full, y_pred_full, t_fit_total,
                        models_for_size, predict_fns[0], h_test, len(h_train)
                    )
                    metrics["method"] = method
                    metrics["step"] = step
                    metrics["mode"] = mode
                    all_rows.append(metrics)
                    print(f"   ✓ {method:<30} R²(T)={metrics['R2_T']:.4f}  "
                          f"MAE(T)={metrics['MAE_T']:.3f}  "
                          f"MAE(P)={metrics['MAE_P']:.1f}")
                except Exception as e:
                    print(f"   ✗ {method}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    print(f"\n✅ Длинная таблица: {os.path.join(RESULTS_DIR, 'raw_results.csv')}")

    # ============================================================
    # Pivot-таблицы: для каждой метрики × каждого режима
    # ============================================================
    metrics_list = [
        "MAE_T", "MAE_P", "MAE_rho", "MAE_a", "MAE_avg",
        "RMSE_T", "RMSE_P", "RMSE_rho", "RMSE_a", "RMSE_avg",
        "R2_T", "R2_P", "R2_rho", "R2_a", "R2_avg",
        "MAPE_T", "MAPE_P", "MAPE_rho", "MAPE_a",
        "MaxErr_T", "MaxErr_P", "MaxErr_rho", "MaxErr_a",
        "time_1point_us", "model_size_kb", "train_time_s",
        "physics_total", "physics_hydro", "physics_gas", "physics_sound",
    ]

    print("\n=== Создание pivot-таблиц ===")
    for mode in MODES:
        for metric in metrics_list:
            sub = df[df["mode"] == mode]
            if metric not in sub.columns or sub[metric].isna().all():
                continue
            pivot = sub.pivot_table(index="method", columns="step",
                                     values=metric, aggfunc="mean")
            pivot = pivot.reindex([m for m in methods if m in pivot.index])
            pivot.to_csv(os.path.join(TABLES_DIR, f"{metric}__{mode}.csv"))
        print(f"  ✓ mode = {mode}: {len(metrics_list)} таблиц")

    # ============================================================
    # Три ключевые таблицы — на экран
    # ============================================================
    for metric in ["MAE_T", "R2_a", "physics_total"]:
        for mode in MODES:
            sub = df[df["mode"] == mode]
            if metric not in sub.columns or sub[metric].isna().all():
                continue
            print(f"\n{'='*70}\n{metric} ({mode})\n{'='*70}")
            p = sub.pivot_table(index="method", columns="step",
                                 values=metric, aggfunc="mean")
            print(p.round(4).to_string())

    print(f"\n✅ Готово. Все таблицы в {TABLES_DIR}")