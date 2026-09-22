"""
Загрузка CSV-файлов ISO 5878 и построение датасетов 1D, 2D, 3D, 4D.

Запуск:
    python experiments/load_iso5878.py

Результат:
    data/datasets/1d.csv
    data/datasets/2d.csv
    data/datasets/3d.csv
    data/datasets/4d.csv
"""
import os
import sys
import numpy as np
import pandas as pd

# ---------- Пути ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISO_DIR = os.path.join(PROJECT_ROOT, "data", "iso5878")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Конфигурация ----------
PROFILES = [
    ("A02_lat15_annual.csv",     15, "annual",       False),
    ("A03_lat30_winter.csv",     30, "winter",       False),
    ("A04_lat30_summer.csv",     30, "summer",       False),
    ("A05_lat45_winter.csv",     45, "winter",       False),
    ("A06_lat45_summer.csv",     45, "summer",       False),
    ("A07_lat60_winter.csv",     60, "winter",       False),
    ("A08_lat60_winter_cold.csv",60, "winter_cold",  True),
    ("A09_lat60_winter_warm.csv",60, "winter_warm",  True),
    ("A10_lat60_summer.csv",     60, "summer",       False),
    ("A11_lat80_winter.csv",     80, "winter",       False),
    ("A12_lat80_winter_cold.csv",80, "winter_cold",  True),
    ("A13_lat80_winter_warm.csv",80, "winter_warm",  True),
    ("A14_lat80_summer.csv",     80, "summer",       False),
]

SEASONS = ["winter", "summer"]
LATS = [15, 30, 45, 60, 80]

# Физические константы
R_DRY = 287.053      # Дж/(кг·К) — удельная газовая постоянная сухого воздуха
R_VAP = 461.5        # Дж/(кг·К) — удельная газовая постоянная водяного пара
GAMMA = 1.4


# ============================================================
# 1. Загрузка профилей
# ============================================================
def load_profiles():
    """Загружает все 13 CSV-профилей в один DataFrame."""
    frames = []
    for fname, lat, season, is_strat_regime in PROFILES:
        path = os.path.join(ISO_DIR, "profiles", fname)
        if not os.path.exists(path):
            print(f"⚠ Пропущен: {fname}")
            continue
        df = pd.read_csv(path).dropna(subset=["temperature_K"])
        df["latitude_deg"] = lat
        df["season"] = season
        df["strat_regime"] = season if is_strat_regime else "mean"
        df["pressure_Pa"] = df["pressure_hPa"] * 100.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_surface():
    """Граничные условия на уровне моря."""
    path = os.path.join(ISO_DIR, "surface", "A01_surface_conditions.csv")
    return pd.read_csv(path)


def load_humidity():
    """Влажностные характеристики (Table C.1)."""
    path = os.path.join(ISO_DIR, "humidity", "C01_humidity_mixing_ratio.csv")
    df = pd.read_csv(path).dropna(subset=["r_g_per_kg"])
    return df


# ============================================================
# 2. Скорость звука и плотность
# ============================================================
def compute_speed_of_sound(T):
    """a = sqrt(gamma * R * T)."""
    return np.sqrt(GAMMA * R_DRY * T)


def compute_density_dry(P, T):
    """ρ = P / (R_dry * T) — для сухого воздуха."""
    return P / (R_DRY * T)


def compute_density_humid(P, T, e_prime_Pa):
    """
    Плотность влажного воздуха:
    ρ = (P - e')/(R_dry·T) + e'/(R_vap·T)
    """
    return (P - e_prime_Pa) / (R_DRY * T) + e_prime_Pa / (R_VAP * T)


# ============================================================
# 3. Построение 1D датасета (ГОСТ 4401-81)
# ============================================================
def build_1d():
    """Базовый 1D: f(h) → (T, P, ρ, a). Берём из эталонного ГОСТ 4401-81."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
    from app.models.isa_core import standard_atmosphere

    heights = np.arange(0, 20001, 100)
    rows = []
    for h in heights:
        T, P, rho, a = standard_atmosphere(float(h))
        rows.append({"h": h, "T": T, "P": P, "rho": rho, "a": a})
    return pd.DataFrame(rows)


# ============================================================
# 4. Построение 2D датасета: (h, φ) — усреднение по сезону
# ============================================================
def build_2d(profiles):
    """
    Усредняем по сезону. Для 15° только annual — берём его.
    Для 60° и 80° усредняем winter и summer (без cold/warm).
    """
    df = profiles[profiles["strat_regime"] == "mean"].copy()

    # Для 15° сезон уже annual — оставляем
    # Для остальных — усредняем winter и summer
    result = []
    for lat in LATS:
        sub = df[df["latitude_deg"] == lat]
        if sub["season"].nunique() == 1:
            # Только annual (15°)
            grouped = sub.groupby("altitude_m").agg({
                "H_geopotential_m": "mean",
                "temperature_K": "mean",
                "pressure_Pa": "mean",
                "density_kg_m3": "mean",
            }).reset_index()
        else:
            # Усредняем winter и summer
            grouped = sub.groupby("altitude_m").agg({
                "H_geopotential_m": "mean",
                "temperature_K": "mean",
                "pressure_Pa": "mean",
                "density_kg_m3": "mean",
            }).reset_index()
        grouped["latitude_deg"] = lat
        result.append(grouped)

    out = pd.concat(result, ignore_index=True)
    out = out.rename(columns={"altitude_m": "h"})
    out["a"] = compute_speed_of_sound(out["temperature_K"])
    out["rho"] = compute_density_dry(out["pressure_Pa"], out["temperature_K"])
    return out[["h", "latitude_deg", "temperature_K", "pressure_Pa",
                "density_kg_m3", "a"]].rename(columns={
        "temperature_K": "T", "pressure_Pa": "P", "density_kg_m3": "rho"
    })


# ============================================================
# 5. Построение 3D датасета: (h, φ, t)
# ============================================================
def build_3d(profiles):
    """Без усреднения по сезону. Только mean-режим."""
    df = profiles[profiles["strat_regime"] == "mean"].copy()
    df = df.rename(columns={
        "altitude_m": "h",
        "temperature_K": "T",
        "pressure_Pa": "P",
        "density_kg_m3": "rho",
    })
    df["a"] = compute_speed_of_sound(df["T"])
    return df[["h", "latitude_deg", "season", "T", "P", "rho", "a"]]


# ============================================================
# 6. Построение 4D датасета: (h, φ, t, w)
# ============================================================
def interp_humidity_by_lat(humidity, target_lats=[15, 30, 45, 60, 80]):
    """
    Интерполирует влажность с широт C.01 (10, 30, 50, 70) на широты A
    (15, 30, 45, 60, 80). Для 80° — экстраполяция от 70.
    """
    result = []
    for lat in target_lats:
        for season in ["January", "July"]:
            sub = humidity[(humidity["season"] == season)].copy()
            sub = sub.sort_values("latitude_deg")

            for h_km in sorted(sub["altitude_km"].unique()):
                sub_h = sub[sub["altitude_km"] == h_km]
                lats_c = sub_h["latitude_deg"].values
                r_vals = sub_h["r_g_per_kg"].values
                e_vals = sub_h["e_prime_hPa"].values

                # Линейная интерполяция / экстраполяция
                r_interp = np.interp(lat, lats_c, r_vals)
                e_interp = np.interp(lat, lats_c, e_vals)

                result.append({
                    "latitude_deg": lat,
                    "season": season,
                    "altitude_km": h_km,
                    "r_g_per_kg": r_interp,
                    "e_prime_hPa": e_interp,
                })
    return pd.DataFrame(result)


def build_4d(profiles, humidity):
    """
    4D: (h, φ, t, w). w ∈ [0, 1] — относительная влажность.
    Ограничение: h ≤ 10 000 м, так как данные по влажности
    в ISO 5878 заданы только до этой высоты.
    """
    hum_interp = interp_humidity_by_lat(humidity)
    season_map = {"winter": "January", "summer": "July", "annual": "Annual"}

    base = build_3d(profiles)
    # ФИЛЬТР: только h ≤ 10 км
    base = base[base["h"] <= 10000].copy()

    w_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []

    for _, row in base.iterrows():
        h = row["h"]
        lat = row["latitude_deg"]
        season_a = row["season"]

        if season_a not in season_map:
            continue
        season_c = season_map[season_a]

        h_km = h / 1000.0
        hum_row = hum_interp[
            (hum_interp["latitude_deg"] == lat) &
            (hum_interp["season"] == season_c)
        ]

        if len(hum_row) > 0:
            hum_row = hum_row.iloc[
                (hum_row["altitude_km"] - h_km).abs().argmin()
            ]
            r_max = hum_row["r_g_per_kg"]
            e_max = hum_row["e_prime_hPa"] * 100.0
        else:
            r_max = 0.0
            e_max = 0.0

        for w in w_grid:
            e_prime = w * e_max
            rho_humid = compute_density_humid(row["P"], row["T"], e_prime)
            rows.append({
                "h": h,
                "latitude_deg": lat,
                "season": season_a,
                "w": w,
                "r_g_per_kg": w * r_max,
                "T": row["T"],
                "P": row["P"],
                "rho": rho_humid,
                "a": compute_speed_of_sound(row["T"]),
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. Главный пайплайн
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Загрузка и построение датасетов ISO 5878")
    print("=" * 60)

    print("\n▶ Загрузка профилей...")
    profiles = load_profiles()
    print(f"  Загружено: {len(profiles)} строк")

    print("\n▶ Загрузка влажности...")
    humidity = load_humidity()
    print(f"  Загружено: {len(humidity)} строк")

    print("\n▶ Построение 1D (ГОСТ 4401-81)...")
    df1 = build_1d()
    df1.to_csv(os.path.join(OUT_DIR, "1d.csv"), index=False)
    print(f"  ✅ 1D: {len(df1)} строк × {len(df1.columns)} столбцов")

    print("\n▶ Построение 2D (h, φ)...")
    df2 = build_2d(profiles)
    df2.to_csv(os.path.join(OUT_DIR, "2d.csv"), index=False)
    print(f"  ✅ 2D: {len(df2)} строк × {len(df2.columns)} столбцов")

    print("\n▶ Построение 3D (h, φ, t)...")
    df3 = build_3d(profiles)
    df3.to_csv(os.path.join(OUT_DIR, "3d.csv"), index=False)
    print(f"  ✅ 3D: {len(df3)} строк × {len(df3.columns)} столбцов")

    print("\n▶ Построение 4D (h, φ, t, w)...")
    df4 = build_4d(profiles, humidity)
    df4.to_csv(os.path.join(OUT_DIR, "4d.csv"), index=False)
    print(f"  ✅ 4D: {len(df4)} строк × {len(df4.columns)} столбцов")

    print("\n" + "=" * 60)
    print("Итоговая статистика:")
    print("=" * 60)
    for name, df in [("1D", df1), ("2D", df2), ("3D", df3), ("4D", df4)]:
        print(f"\n{name}:")
        print(df.describe().round(3).to_string())