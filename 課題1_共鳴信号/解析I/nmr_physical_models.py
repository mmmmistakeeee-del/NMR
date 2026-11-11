"""Exploratory CW-NMR fits with explicit measurement broadening.

Run: python -B nmr_physical_models.py [two-column-data.dat]
Input frequencies are Hz; internal frequency offsets and widths are kHz.
All fits use the same data, unweighted least squares and a linear baseline.
Finite modulation assumes sinusoidal field modulation and quasi-static response.
The fitted modulation amplitude is a PEAK frequency excursion, not peak-to-peak.
Voigt broadening assumes a Gaussian distribution of local resonance frequencies.
Neither model includes sweep lag, Bloch dynamics or an independently calibrated
noise covariance. Widths are effective fit parameters, not validated T2 values.
"""

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import wofz


def lorentz_derivative(x, gamma):
    """ローレンツ吸収の周波数微分。xと半値半幅gammaの単位はkHz。"""
    return -2 * gamma * x / (np.pi * (x*x + gamma*gamma)**2)


def modulated_lorentz(x, gamma, excursion):
    """First cosine harmonic divided by excursion; tends to L'(x).

    日本語の案内: 正弦波磁場変調の基本波を、ローレンツ吸収から計算する。
    excursionは変調の片振幅[kHz]。出力は基本波/excursionで、
    excursionが小さいとローレンツ吸収の微分になる。

    For z=x+i*gamma, the analytic integral uses the branch of sqrt(z^2-a^2)
    that approaches z in the upper half plane. Dividing by a simply changes
    the free amplitude; it does not change the fitted lineshape.
    """
    if excursion < 1e-4 * gamma:
        return lorentz_derivative(x, gamma)
    z = np.asarray(x) + 1j * gamma
    root = np.sqrt(z-excursion) * np.sqrt(z+excursion)
    # z/root - 1 = a^2 / (root*(z+root)); avoids small-a cancellation.
    return 2 / np.pi * np.imag(1 / (root * (z+root)))


def voigt_derivative(x, gamma, sigma):
    """Voigt吸収の周波数微分。x、gamma、sigmaはkHz。

    gammaはローレンツ半値半幅、sigmaはガウス分布の標準偏差。
    SciPyの複素誤差関数wofzを使い、畳み込みを毎回直接積分せず計算する。
    """
    if sigma < 1e-4 * max(gamma, 1e-10):
        return lorentz_derivative(x, gamma)
    z = (x + 1j*gamma) / (sigma*np.sqrt(2))
    wp = -2*z*wofz(z) + 2j/np.sqrt(np.pi)
    return np.real(wp)/(2*sigma*sigma*np.sqrt(np.pi))


def shape(kind, x, q):
    center, width = q[:2]
    u = x-center
    if kind == "lorentz":
        return lorentz_derivative(u, width)
    if kind == "gaussian":
        return -u*np.exp(-0.5*(u/width)**2)/(np.sqrt(2*np.pi)*width**3)
    if kind == "modulated_lorentz":
        return modulated_lorentz(u, width, q[2])
    if kind == "voigt":
        return voigt_derivative(u, width, q[2])
    raise ValueError(kind)


def fit_model(kind, x, y):
    # Eliminate linear amplitude, offset and slope by least squares at each step.
    def solve(q):
        matrix = np.column_stack([shape(kind, x, q), np.ones(len(x)), x])
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        return matrix@beta, beta

    center_bounds = (float(x.min()), float(x.max()))
    if kind in ("lorentz", "gaussian"):
        starts = [[0., width] for width in (0.5, 1.2, 2.)]
        bounds = ([center_bounds[0], 0.001], [center_bounds[1], 30.])
    elif kind == "modulated_lorentz":
        starts = [[0., g, a] for g, a in ((1.7, .01), (1., 1.), (.6, 2.), (.2, 2.))]
        bounds = ([center_bounds[0], 0.001, 0.], [center_bounds[1], 30., 30.])
    else:
        starts = [[0., g, s] for g, s in ((0., 1.2), (.3, 1.), (1., .5), (1.7, .05))]
        bounds = ([center_bounds[0], 0., .001], [center_bounds[1], 30., 30.])
    candidates = [least_squares(lambda q: solve(q)[0]-y, start, bounds=bounds,
                                max_nfev=3000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                  for start in starts]
    valid = [result for result in candidates if result.success]
    if not valid:
        raise RuntimeError(f"No converged fit for {kind}")
    result = min(valid, key=lambda r: np.sum(r.fun*r.fun))
    prediction, beta = solve(result.x)
    return result.x, beta, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default="20261020_KITtest01.dat")
    args = parser.parse_args()
    src = Path(args.file)
    data = np.loadtxt(src)
    f, y = data[:, 0], data[:, 1]
    if not np.isfinite(data).all() or np.any(np.diff(f) <= 0):
        raise ValueError("Expected finite data in strictly increasing frequency order")
    ref = float((f[np.argmax(y)]+f[np.argmin(y)])/2)
    x = (f-ref)/1000
    core = abs(x) < 5
    kinds = ["lorentz", "gaussian", "modulated_lorentz", "voigt"]
    fits, rows, details = {}, [], {}
    for kind in kinds:
        q, beta, pred = fit_model(kind, x, y)
        r = y-pred
        sse = float(np.sum(r*r))
        k = len(q)+len(beta)
        # Additional unknown iid noise variance contributes one parameter.
        aic = len(y)*np.log(sse/len(y)) + 2*(k+1)
        row = dict(model=kind, mean_parameters=k, f0_MHz=(ref+1000*q[0])/1e6,
                   gamma_kHz=q[1] if kind != "gaussian" else np.nan,
                   sigma_kHz=q[1] if kind == "gaussian" else (q[2] if kind == "voigt" else np.nan),
                   modulation_peak_kHz=q[2] if kind == "modulated_lorentz" else np.nan,
                   RMS=np.sqrt(sse/len(y)), core_RMS=np.sqrt(np.mean(r[core]**2)),
                   core_lag1=np.corrcoef(r[core][:-1],r[core][1:])[0,1],
                   iid_gaussian_AIC=aic)
        rows.append(row)
        fits[kind] = (q, beta, pred)
        details[kind] = dict(shape_parameters=q.tolist(), linear_parameters=beta.tolist())
    table = pd.DataFrame(rows)
    table["iid_delta_AIC"] = table.iid_gaussian_AIC-table.iid_gaussian_AIC.min()
    prefix = src.with_name(src.stem+"_physical_models")
    table.to_csv(str(prefix)+".csv", index=False)
    details["assumptions"] = {
        "frequency_reference_Hz": ref, "core_points": int(sum(core)),
        "fit": "Unweighted least squares; linear baseline; identical full data",
        "modulation": "Quasi-static sinusoidal field modulation; peak excursion",
        "noise": "AIC assumes independent equal-variance Gaussian noise; diagnostic only",
        "limitations": "No sweep lag or finite modulation frequency dynamics; no T2 inference",
    }
    Path(str(prefix)+".json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("\nWindow sensitivity: refit modulated Lorentz with a free linear baseline")
    for radius in (5, 10, 15, 20, 30):
        mask = abs(x)<radius
        q, beta, pred = fit_model("modulated_lorentz", x[mask], y[mask])
        print(f"+/-{radius} kHz: gamma={q[1]:.6f}, modulation_peak={q[2]:.6f} kHz, "
              f"RMS={np.sqrt(np.mean((y[mask]-pred)**2)):.6f}")
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    mask = abs(x)<7
    xx = np.linspace(-7,7,1800)
    ax.scatter(x[mask],y[mask],s=17,c="black",alpha=.55,label="Measured")
    labels = {"lorentz": "Lorentz derivative", "gaussian": "Gaussian derivative",
              "modulated_lorentz": "Lorentz + finite modulation", "voigt": "Voigt derivative"}
    for kind, color, style in zip(kinds, ["#777777", "#d48000", "#1464b0", "#439c68"],
                                  ["--", "--", "-", ":"]):
        q,beta,pred=fits[kind]
        ax.plot(xx, beta[0]*shape(kind,xx,q)+beta[1]+beta[2]*xx,
                color=color,ls=style,lw=2,label=labels[kind])
        ar.plot(x[mask],(y-pred)[mask],color=color,ls=style,lw=1,alpha=.9)
    ax.set_ylabel("Signal (original data units)")
    ax.set_title("CW-NMR: intrinsic lineshape and measurement broadening")
    ax.legend(fontsize=9)
    ar.axhline(0,color="black",lw=.6)
    ar.set_xlabel(f"Frequency offset from {ref/1e6:.6f} MHz (kHz)")
    ar.set_ylabel("Measured - fit")
    for axis in (ax,ar):
        axis.grid(alpha=.2)
    fig.text(.5,.01,"Finite modulation fit assumes quasi-static response; fitted widths are effective parameters.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,1))
    fig.savefig(str(prefix)+".png",dpi=170)
    plt.close(fig)
    print("\nSaved:",str(prefix)+".{csv,json,png}")


if __name__ == "__main__":
    main()
