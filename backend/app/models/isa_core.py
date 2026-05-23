import numpy as np

def standard_atmosphere(h: float):
    """
    Возвращает (T, P, ρ, a) для высоты h (м) в диапазоне 0..20000 м.
    Соответствует ГОСТ 4401-81 / ISO 2533.
    """
    T0 = 288.15
    P0 = 101325.0
    g = 9.80665
    R = 287.053
    if h <= 11000:
        T = T0 - 0.0065 * h
        P = P0 * (T / T0) ** (g / (0.0065 * R))
    elif h <= 20000:
        h11 = 11000
        T11 = 216.65
        _, P11, _, _ = standard_atmosphere(11000)
        P = P11 * np.exp(-g / (R * T11) * (h - h11))
        T = T11
    else:
        raise ValueError("Высота вне диапазона 0..20000 м")
    rho = P / (R * T)
    a = np.sqrt(1.4 * R * T)
    return T, P, rho, a