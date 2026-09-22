"""
Визуализация результатов обучения моделей МСА.

Запуск:
    python experiments/visualize_results.py

Результат:
    results/figures/*.png
    results/figures/*.pdf
    results/figures/*.svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"  # поддерживает кириллицу
plt.rcParams["font.size"] = 11

# ---------- Пути ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "training")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Загрузка ----------
df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))

DATASETS = ["1d", "2d", "3d", "4d"]
DATASET_LABELS = {"1d": "1D", "2d": "2D", "3d": "3D", "4d": "4D"}

# ---------- Группы и цвета ----------
GROUP_COLORS = {
    "Классическая интерполяция": "#d62728",  # красный
    "ML-регрессия":              "#1f77b4",  # синий
    "Baseline":                  "#7f7f7f",  # серый
}

METHOD_GROUPS = {
    "Классическая интерполяция": [
        "Линейная интерполяция", "Кубический сплайн", "PCHIP", "RBF-интерполяция",
    ],
    "ML-регрессия": [
        "MLP", "GPR", "GradientBoosting", "RandomForest", "SVR",
    ],
    "Baseline": [
        "LinearRegression", "PolynomialFeatures(5)+LR",
    ],
}

METHOD_TO_GROUP = {}
for group, methods in METHOD_GROUPS.items():
    for m in methods:
        METHOD_TO_GROUP[m] = group

# Стили линий и маркеров для ключевых методов
KEY_METHODS = {
    "Кубический сплайн":       ("o", "-",  "Кубический сплайн"),
    "RBF-интерполяция":         ("s", "--", "RBF-интерполяция"),
    "MLP":                     ("^", "-",  "MLP"),
    "GPR":                     ("D", "-",  "GPR"),
    "GradientBoosting":         ("v", "--", "Gradient Boosting"),
    "RandomForest":             ("P", "--", "Random Forest"),
    "Линейная интерполяция":    ("x", ":",  "Линейная интерполяция"),
    "PCHIP":                    ("+", ":",  "PCHIP"),
}


def save_figure(fig, name):
    """Сохраняет график во всех форматах."""
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {name}.png / .pdf / .svg")


def pivot_metric(metric):
    """Строит таблицу: строки — методы, столбцы — датасеты."""
    return df.pivot_table(index="method", columns="dataset",
                           values=metric, aggfunc="mean")


# ============================================================
# ГРАФИК 1: MAE vs Размерность
# ============================================================
def plot_mae_vs_dim():
    fig, ax = plt.subplots(figsize=(8, 5))
    p = pivot_metric("MAE_avg")

    for method, (marker, ls, label) in KEY_METHODS.items():
        if method not in p.index:
            continue
        y_vals = []
        x_vals = []
        for i, ds in enumerate(DATASETS):
            if ds in p.columns and not pd.isna(p.loc[method, ds]):
                x_vals.append(i + 1)
                y_vals.append(p.loc[method, ds])
        if not y_vals:
            continue
        group = METHOD_TO_GROUP.get(method, "ML-регрессия")
        ax.plot(x_vals, y_vals, marker=marker, linestyle=ls,
                label=label, color=GROUP_COLORS[group], markersize=8,
                linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Размерность задачи", fontsize=12)
    ax.set_ylabel("MAE (усреднённая по 4 параметрам)", fontsize=12)
    ax.set_title("Точность методов при росте размерности", fontsize=13)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1D", "2D", "3D", "4D"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "01_mae_vs_dim")


# ============================================================
# ГРАФИК 2: Размер модели vs Размерность (ГЛАВНЫЙ)
# ============================================================
def plot_size_vs_dim():
    fig, ax = plt.subplots(figsize=(8, 5))
    p = pivot_metric("model_size_kb")

    for method, (marker, ls, label) in KEY_METHODS.items():
        if method not in p.index:
            continue
        y_vals = []
        x_vals = []
        for i, ds in enumerate(DATASETS):
            if ds in p.columns and not pd.isna(p.loc[method, ds]):
                x_vals.append(i + 1)
                y_vals.append(p.loc[method, ds])
        if not y_vals:
            continue
        group = METHOD_TO_GROUP.get(method, "ML-регрессия")
        ax.plot(x_vals, y_vals, marker=marker, linestyle=ls,
                label=label, color=GROUP_COLORS[group], markersize=8,
                linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Размерность задачи", fontsize=12)
    ax.set_ylabel("Размер модели, КБ", fontsize=12)
    ax.set_title("Рост размера модели при увеличении размерности", fontsize=13)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1D", "2D", "3D", "4D"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Аннотация ключевого вывода
    ax.annotate("MLP: размер\nпочти не растёт",
                xy=(4, 240), xytext=(2.5, 2000),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))

    save_figure(fig, "02_size_vs_dim")


# ============================================================
# ГРАФИК 3: Время предсказания vs Размерность
# ============================================================
def plot_time_vs_dim():
    fig, ax = plt.subplots(figsize=(8, 5))
    p = pivot_metric("time_1point_us")

    for method, (marker, ls, label) in KEY_METHODS.items():
        if method not in p.index:
            continue
        y_vals = []
        x_vals = []
        for i, ds in enumerate(DATASETS):
            if ds in p.columns and not pd.isna(p.loc[method, ds]):
                x_vals.append(i + 1)
                y_vals.append(p.loc[method, ds])
        if not y_vals:
            continue
        group = METHOD_TO_GROUP.get(method, "ML-регрессия")
        ax.plot(x_vals, y_vals, marker=marker, linestyle=ls,
                label=label, color=GROUP_COLORS[group], markersize=8,
                linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Размерность задачи", fontsize=12)
    ax.set_ylabel("Время предсказания 1 точки, мкс", fontsize=12)
    ax.set_title("Скорость инференса при росте размерности", fontsize=13)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1D", "2D", "3D", "4D"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "03_time_vs_dim")


# ============================================================
# ГРАФИК 4: Scatter — MAE vs Размер (4D)
# ============================================================
def plot_scatter_4d():
    fig, ax = plt.subplots(figsize=(8, 6))

    p_mae = pivot_metric("MAE_avg")
    p_size = pivot_metric("model_size_kb")
    p_time = pivot_metric("time_1point_us")

    for method in p_mae.index:
        if "4d" not in p_mae.columns:
            continue
        mae = p_mae.loc[method, "4d"]
        size = p_size.loc[method, "4d"] if "4d" in p_size.columns else np.nan
        t = p_time.loc[method, "4d"] if "4d" in p_time.columns else np.nan

        if pd.isna(mae) or pd.isna(size):
            continue

        group = METHOD_TO_GROUP.get(method, "ML-регрессия")
        # Размер маркера ~ log(время предсказания)
        if not pd.isna(t):
            marker_size = 50 + 30 * np.log10(max(t, 1))
        else:
            marker_size = 100

        ax.scatter(size, mae, s=marker_size,
                   color=GROUP_COLORS[group], alpha=0.75,
                   edgecolors="black", linewidths=1.2)
        ax.annotate(method, (size, mae),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Размер модели, КБ (log)", fontsize=12)
    ax.set_ylabel("MAE на 4D (log)", fontsize=12)
    ax.set_title("Компромисс «точность — память» на 4D-задаче", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")

    # Легенда по группам
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS["Классическая интерполяция"],
               markersize=10, label="Классическая интерполяция"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS["ML-регрессия"],
               markersize=10, label="ML-регрессия"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS["Baseline"],
               markersize=10, label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    # Аннотация оптимальной зоны
    ax.annotate("Оптимальная зона\n(мало памяти, приемлемая точность)",
                xy=(500, 50), xytext=(100, 3000),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))

    save_figure(fig, "04_scatter_mae_size_4d")


# ============================================================
# ГРАФИК 5: Bar chart MAE по 4D (для одной картинки)
# ============================================================
def plot_bar_mae_4d():
    fig, ax = plt.subplots(figsize=(10, 5))
    p = pivot_metric("MAE_avg")
    if "4d" not in p.columns:
        return

    mae_4d = p["4d"].dropna().sort_values()
    colors = [GROUP_COLORS[METHOD_TO_GROUP.get(m, "ML-регрессия")]
              for m in mae_4d.index]

    bars = ax.barh(range(len(mae_4d)), mae_4d.values, color=colors,
                    edgecolor="black", linewidth=0.8)

    ax.set_yticks(range(len(mae_4d)))
    ax.set_yticklabels(mae_4d.index, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("MAE на 4D (log)", fontsize=12)
    ax.set_title("Точность методов на 4D-задаче", fontsize=13)
    ax.grid(True, axis="x", alpha=0.3, which="both")

    # Подписи значений
    for i, v in enumerate(mae_4d.values):
        ax.text(v * 1.1, i, f"{v:.2f}", va="center", fontsize=10)

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GROUP_COLORS["Классическая интерполяция"],
               lw=8, label="Классическая интерполяция"),
        Line2D([0], [0], color=GROUP_COLORS["ML-регрессия"],
               lw=8, label="ML-регрессия"),
        Line2D([0], [0], color=GROUP_COLORS["Baseline"],
               lw=8, label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    save_figure(fig, "05_bar_mae_4d")


# ============================================================
# ГРАФИК 6: Overfit ratio vs Размерность
# ============================================================
def plot_overfit_vs_dim():
    fig, ax = plt.subplots(figsize=(8, 5))
    p = pivot_metric("overfit_ratio")

    for method, (marker, ls, label) in KEY_METHODS.items():
        if method not in p.index:
            continue
        y_vals = []
        x_vals = []
        for i, ds in enumerate(DATASETS):
            if ds in p.columns and not pd.isna(p.loc[method, ds]):
                x_vals.append(i + 1)
                y_vals.append(p.loc[method, ds])
        if not y_vals:
            continue
        group = METHOD_TO_GROUP.get(method, "ML-регрессия")
        ax.plot(x_vals, y_vals, marker=marker, linestyle=ls,
                label=label, color=GROUP_COLORS[group], markersize=8,
                linewidth=2)

    ax.axhline(1.0, color="green", linestyle=":", linewidth=1.5,
                label="Идеал (overfit = 1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("Размерность задачи", fontsize=12)
    ax.set_ylabel("Overfit ratio (MAE_test / MAE_train)", fontsize=12)
    ax.set_title("Устойчивость методов: переобучение", fontsize=13)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1D", "2D", "3D", "4D"])
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "06_overfit_vs_dim")


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Создание визуализаций результатов")
    print("=" * 60)

    print("\n▶ График 1: MAE vs размерность")
    plot_mae_vs_dim()

    print("\n▶ График 2: Размер модели vs размерность")
    plot_size_vs_dim()

    print("\n▶ График 3: Время предсказания vs размерность")
    plot_time_vs_dim()

    print("\n▶ График 4: Scatter MAE vs размер (4D)")
    plot_scatter_4d()

    print("\n▶ График 5: Bar chart MAE на 4D")
    plot_bar_mae_4d()

    print("\n▶ График 6: Overfit ratio vs размерность")
    plot_overfit_vs_dim()

    print(f"\n✅ Все графики сохранены в {FIG_DIR}/")
    print("   Форматы: PNG (300 dpi), PDF, SVG")