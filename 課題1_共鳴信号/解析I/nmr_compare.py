"""Compare derivative-Lorentzian and derivative-Gaussian line shapes for CW-NMR.

Usage:
    python3 nmr_compare.py 20261020_KITtest01.dat --unit Hz

Requires: numpy, pandas, scipy, matplotlib.
Data: 2 columns of frequency and lock-in signal, whitespace or comma separated.
This comparison assumes small modulation amplitude and a quasi-static scan.
It does NOT correct instrument sweep lag, field inhomogeneity or modulation broadening.
"""
#%%
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")     # file output; suitable for WSL/headless computers
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def response(x, center, width, amp, offset, slope, kind="lorentz"):
    """x/center/width in kHz, amplitude/signal/offset in signal units."""
    u = (x-center) / width
    if kind == "lorentz":
        shape = u / (1 + u*u)**2
    elif kind == "gaussian":
        shape = u * np.exp(-0.5*u*u)
    else:
        raise ValueError(kind)
    return offset + slope*x + amp*shape


def load_data(path, unit):
    raw = pd.read_csv(path, sep=r"[,\s]+", engine="python", header=None,
                      comment="#", dtype=str)
    if raw.shape[1] < 2:
        raise ValueError("Need 2 columns: frequency and signal")
    arr = raw.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    if len(arr) < 20:
        raise ValueError("Need at least 20 numeric data points")
    arr = arr.copy()  # pandas may return a read-only NumPy view
    arr[:, 0] *= {"Hz": 1, "kHz": 1000, "MHz": 1e6}[unit]
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0], arr[:, 1]


def fit(x, y, center0, kind):
    peak_hi, peak_lo = x[np.argmax(y)], x[np.argmin(y)]
    peak_sep = abs(peak_lo - peak_hi)
    width0 = max(peak_sep * (np.sqrt(3) / 2 if kind == "lorentz" else .5), .15)
    shape_factor = 9/(16*np.sqrt(3)) if kind == "lorentz" else np.exp(-.5)
    # model's peak magnitude at u=1/sqrt(3) (lorentz) or u=1 (gaussian)
    a0 = np.sign(peak_hi-peak_lo) * max(np.ptp(y)/2/shape_factor, .1)
    initial = [center0, width0, a0, float(np.median(y)), 0.0]
    low = [float(x.min()), .01, -np.inf, -np.inf, -np.inf]
    high = [float(x.max()), np.inf, np.inf, np.inf, np.inf]
    params, covariance = curve_fit(
        lambda x, *p: response(x, *p, kind=kind),
        x, y, p0=initial, bounds=(low, high), maxfev=100000)
    return params, covariance


def crossing(f, y, near_hz):
    indices = np.flatnonzero((y[:-1]*y[1:] < 0))
    if not len(indices):
        return float("nan")
    k = indices[np.argmin(abs((f[indices] + f[indices+1])/2 - near_hz))]
    return float(f[k] - y[k] * (f[k+1]-f[k]) / (y[k+1]-y[k]))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("file", help="two-column NMR data file")
    cli.add_argument("--unit", choices=["Hz", "kHz", "MHz"], default="Hz")
    cli.add_argument("--window", type=float, default=15.0,
                     help="half-width in kHz of plotted close-up (default 15)")
    args = cli.parse_args()
    src = Path(args.file)
    f, y = load_data(src, args.unit)
    ref = float((f[np.argmax(y)] + f[np.argmin(y)])/2)
    x = (f-ref) / 1000  # kHz; fit with small numbers
    center0 = (x[np.argmax(y)] + x[np.argmin(y)])/2
    fits = {}
    rows = []
    for kind in ["lorentz", "gaussian"]:
        params, covariance = fit(x, y, center0, kind)
        cent, width, amp, offset, slope = params
        pred = response(x, *params, kind=kind)
        rms = np.sqrt(np.mean((y-pred)**2))
        within = abs(f - (ref+1000*cent)) < 5000
        rms_core = np.sqrt(np.mean((y[within]-pred[within])**2))
        f0 = ref+1000*cent
        pp_khz = 2*width/np.sqrt(3) if kind == "lorentz" else 2*width
        fwhm_khz = 2*width if kind == "lorentz" else 2*np.sqrt(2*np.log(2))*width
        std_hz = 1000*np.sqrt(covariance[0,0]) if covariance[0,0]>=0 else np.nan
        fits[kind] = (params, pred, f0)
        rows.append(dict(model=kind, f0_MHz=f0/1e6,
                         peak_to_peak_kHz=pp_khz,
                         absorption_FWHM_kHz=fwhm_khz,
                         residual_RMS=rms, residual_RMS_core_5kHz=rms_core,
                         formal_center_SE_Hz=std_hz))
    tab = pd.DataFrame(rows)
    raw_zero = crossing(f, y, ref)
    raw_mid = ref
    print(f"Points: {len(y)}, frequency: {f.min()/1e6:.6f} ... {f.max()/1e6:.6f} MHz")
    print(f"Raw two-point zero crossing: {raw_zero/1e6:.8f} MHz (not baseline-corrected)")
    print(f"Raw peak midpoint:          {raw_mid/1e6:.8f} MHz (noise-sensitive)")
    print(tab.to_string(index=False, float_format=lambda z: f"{z:.7f}"))
    print("\nFit-range sensitivity: centers using the same model, slope + offset")
    print("window(+-kHz)   Lorentz(MHz)   Gaussian(MHz)")
    for radius in [10,15,20,25,30]:
        mask = abs(f - ref) <= radius*1000
        cs=[]
        for kind in ["lorentz", "gaussian"]:
            p,_=fit(x[mask],y[mask],center0,kind)
            cs.append((ref+1000*p[0])/1e6)
        print(f" {radius:>3}              {cs[0]:.8f}     {cs[1]:.8f}")

    xx = np.linspace(-args.window, args.window, 1500)
    fig,ax=plt.subplots(figsize=(10,5.6))
    mask = abs(x) <= args.window
    ax.scatter(f[mask]/1e6, y[mask], s=18, alpha=.7, label="Measured")
    for kind, label in [("lorentz", "Derivative Lorentzian"),
                        ("gaussian", "Derivative Gaussian")]:
        p,_,f0=fits[kind]
        ax.plot((ref + xx*1000)/1e6, response(xx,*p,kind=kind),
                linewidth=2, label=f"{label} (f0={f0/1e6:.8f} MHz)")
    ax.axvline(raw_zero/1e6, linestyle=":", alpha=.75, label="Raw zero crossing")
    ax.set(xlabel="Frequency (MHz)",ylabel="Lock-in signal (data units)",
           title="NMR line-shape comparison: measured vs fitted")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    graph1 = src.with_name(src.stem + "_fit_compare.png")
    fig.savefig(graph1, dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.8))
    for kind, label in [("lorentz", "Derivative Lorentzian"),
                        ("gaussian", "Derivative Gaussian")]:
        p,yp,_=fits[kind]
        ax.scatter(f[mask]/1e6, (y-yp)[mask], s=17,
                   alpha=.5, label=f"{label} residual")
    ax.axhline(0, linewidth=1)
    ax.set(xlabel="Frequency (MHz)",ylabel="Measured minus fit",
           title="Fit residuals near resonance")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    graph2=src.with_name(src.stem+"_residual_compare.png")
    fig.savefig(graph2,dpi=180)
    plt.close(fig)

    out_csv=src.with_name(src.stem + "_fit_comparison.csv")
    tab.to_csv(out_csv,index=False)
    print(f"\nSaved: {graph1}\nSaved: {graph2}\nSaved: {out_csv}")
    print("Caution: formal standard errors do not include model, modulation or instrument systematics.")


if __name__ == "__main__":
    main()

# %%
