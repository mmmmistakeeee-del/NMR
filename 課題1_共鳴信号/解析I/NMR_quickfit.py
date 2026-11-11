"""NMR簡易フィット：微分Lorentz / Gaussian / Voigt と不確かさ。

使い方:
    pip install numpy scipy matplotlib
    python nmr_quickfit.py 20261020_KITtest01.dat
    python nmr_quickfit.py 20261020_KITtest01.dat --bootstrap 100

入力: 周波数[Hz] と信号の2列（空白・タブ・カンマ区切り）。
出力: *_quickfit.csv と *_quickfit.png（入力ファイルと同じ場所）。

注意: 標準誤差はモデルを正しいと仮定した統計的な目安。
      周波数校正、有限磁場変調、掃引遅れ等の系統誤差は含まない。
      
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # WSL・端末のみの環境でもPNGを保存
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.special import voigt_profile, wofz

KINDS = ("lorentz", "gaussian", "voigt")
LABELS = {"lorentz": "Derivative Lorentzian",
          "gaussian": "Derivative Gaussian", "voigt": "Derivative Voigt"}


def derivative_shape(t, widths, kind):
    """吸収線形の微分（符号・定数は振幅で吸収）。 t, widths: kHz"""
    if kind == "lorentz":
        gamma = widths[0]
        u = t / gamma
        return u / (1.0 + u*u)**2
    if kind == "gaussian":
        sigma = widths[0]
        u = t / sigma
        return u * np.exp(-0.5*u*u)
    gamma, sigma = widths
    z = (t + 1j*gamma) / (sigma * np.sqrt(2))
    wp = -2*z*wofz(z) + 2j/np.sqrt(np.pi)  # w'(z)
    # Voigt profile の微分: V'(t) [1/kHz^2]
    return np.real(wp) / (2*sigma*sigma*np.sqrt(np.pi))


def model(x, *params, kind):
    """線形背景付き信号: 共鳴位置c + 線形背景b0+b1*x。"""
    nwidths = 2 if kind == "voigt" else 1
    center = params[0]
    widths = params[1:1+nwidths]
    amplitude, offset, slope = params[1+nwidths:]
    return amplitude * derivative_shape(x-center, widths, kind) + offset + slope*x


def get_fits(x, y, start_center):
    """幅を複数の初期値から探索して、残差二乗和の最小解を採用。"""
    initial_peak_sep = max(abs(x[np.argmax(y)] - x[np.argmin(y)]), 0.5)
    amplitude_sign = np.sign(x[np.argmax(y)] - x[np.argmin(y)]) or 1.0
    amp0 = amplitude_sign * max(np.ptp(y), 0.1)
    offset0 = float(np.median(y))
    span = float(np.ptp(x))
    lo, hi = float(x.min()), float(x.max())
    width_cap = max(span, 10.0)
    fits = {}
    for kind in KINDS:
        if kind == "lorentz":
            w = initial_peak_sep * np.sqrt(3)/2
            starts = [[w], [w*.6], [w*1.5]]
        elif kind == "gaussian":
            w = initial_peak_sep / 2
            starts = [[w], [w*.6], [w*1.5]]
        else:
            w = initial_peak_sep / 2
            starts = [[max(w*.5, .03), max(w*.8, .03)],
                      [max(w*.1, .03), max(w, .03)],
                      [max(w, .03), max(w*.2, .03)]]
        best = None
        low = [lo] + [0.005]*len(starts[0]) + [-np.inf]*3
        high = [hi] + [width_cap]*len(starts[0]) + [np.inf]*3
        for ws in starts:
            p0 = [start_center] + ws + [amp0, offset0, 0.0]
            try:
                params, cov = curve_fit(
                    lambda xx, *p: model(xx, *p, kind=kind), x, y,
                    p0=p0, bounds=(low, high), maxfev=25000)
                sse = float(np.sum((y-model(x, *params, kind=kind))**2))
                if np.isfinite(sse) and (best is None or sse < best[0]):
                    best = (sse, params, cov)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
        if best is None:
            raise RuntimeError(f"{kind} のフィットが収束しませんでした")
        fits[kind] = (best[1], best[2])
    return fits


def widths_to_derived(kind, widths):
    """吸収線形の半値全幅 FWHM と微分線形の極大・極小間隔（ともにkHz）。"""
    if kind == "lorentz":
        g = widths[0]
        return 2*g, 2*g/np.sqrt(3)
    if kind == "gaussian":
        s = widths[0]
        return 2*np.sqrt(2*np.log(2))*s, 2*s
    g, s = widths
    v0 = voigt_profile(0, s, g)
    halfwidth = brentq(lambda t: voigt_profile(t, s, g)-v0/2,
                       0, 10*(g+s))
    # 微分Voigtの極値は左右対称。正側の |V'| の極大位置を求める。
    peak = minimize_scalar(
        lambda t: -abs(float(derivative_shape(t, [g, s], "voigt"))),
        bounds=(0, 5*(g+s)), method="bounded").x
    return 2*halfwidth, 2*peak


def derived_errors(kind, p, cov):
    """共分散行列からFWHMと極大・極小間隔の誤差伝播。"""
    n = 2 if kind == "voigt" else 1
    w = np.asarray(p[1:1+n])
    j = np.empty((2, n))
    for i in range(n):
        h = max(w[i]*1e-4, 1e-5)
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        j[:, i] = (np.array(widths_to_derived(kind, wp))
                   - np.array(widths_to_derived(kind, wm))) / (2*h)