"""CW-NMR 280 Hz lock-in data: fit a derivative Lorentzian.

Usage: python nmr_fit.py my_measurement.dat --unit Hz
The first two columns must be frequency and signal voltage, separated by
spaces, tabs, or commas. Headers are accepted. No data are altered.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def derivative_lorentz(x, center, gamma, amplitude, offset, slope):
    """x, center, gamma in kHz, amplitude and offset in data's voltage unit."""
    t = (x - center) / gamma
    return offset + slope * x + amplitude * t / (1.0 + t**2)**2


def main():
    parser = argparse.ArgumentParser(description="Fit a CW-NMR derivative resonance line")
    parser.add_argument("file", help="Two-column measurement file: frequency, signal")
    parser.add_argument("--unit", choices=["Hz", "kHz", "MHz"], default="Hz",
                        help="Frequency unit in the file (default: Hz)")
    args = parser.parse_args()

    # Read whitespace/comma-separated data; ignore non-numeric header rows.
    raw = pd.read_csv(args.file, sep=r"[,\s]+", engine="python", header=None,
                      comment="#", dtype=str)
    if raw.shape[1] < 2:
        raise ValueError("Two numeric columns are required: frequency, signal")
    frame = raw.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 7:
        raise ValueError("At least 7 numeric measurement points are required")

    freq = frame.iloc[:, 0].to_numpy(dtype=float)
    signal = frame.iloc[:, 1].to_numpy(dtype=float)
    freq = freq * {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6}[args.unit]
    order = np.argsort(freq)
    freq, signal = freq[order], signal[order]

    f_ref = float(np.median(freq))
    x = (freq - f_ref) / 1000.0  # kHz: improves numerical stability
    i_high, i_low = np.argmax(signal), np.argmin(signal)
    peak_spacing = abs(x[i_high] - x[i_low])
    if peak_spacing == 0:
        raise ValueError("Peaks could not be separated; inspect the frequency range")

    guess_center = (x[i_high] + x[i_low]) / 2
    guess_gamma = np.sqrt(3) / 2 * peak_spacing
    guess_amp = np.sign(x[i_high] - x[i_low]) * np.ptp(signal) / (9 / (8 * np.sqrt(3)))
    initial = [guess_center, guess_gamma, guess_amp, float(np.median(signal)), 0.0]
    lower = [float(x.min()), 1e-7, -np.inf, -np.inf, -np.inf]
    upper = [float(x.max()), np.inf, np.inf, np.inf, np.inf]
    popt, pcov = curve_fit(derivative_lorentz, x, signal, p0=initial,
                           bounds=(lower, upper), maxfev=30000)
    center, gamma, amplitude, offset, slope = popt

    resonance_hz = f_ref + 1000 * center
    peak_to_peak_khz = 2 * gamma / np.sqrt(3)
    absorption_fwhm_khz = 2 * gamma
    stderr_center_khz = np.sqrt(pcov[0, 0]) if pcov[0, 0] >= 0 else float("nan")
    residual_rms = np.sqrt(np.mean((signal - derivative_lorentz(x, *popt))**2))
    print(f"Resonance f0:          {resonance_hz / 1e6:.8f} MHz")
    print(f"Peak-to-peak width:    {peak_to_peak_khz:.4f} kHz (PDF Fig. 7 definition)")
    print(f"Absorption FWHM:       {absorption_fwhm_khz:.4f} kHz (Lorentz model)")
    print(f"Formal standard error: {stderr_center_khz:.4f} kHz (not systematic error)")
    print(f"Residual RMS:          {residual_rms:.6g} (signal unit)")

    xx = np.linspace(x.min(), x.max(), 1500)
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 figsize=(9, 6))
    ax.scatter(freq / 1e6, signal, s=11, label="Measured data")
    ax.plot((f_ref + 1000 * xx) / 1e6, derivative_lorentz(xx, *popt),
            label="Derivative-Lorentz fit")
    ax.axvline(resonance_hz / 1e6, linestyle="--", label="Resonance f0")
    ax.set_ylabel("Lock-in signal (original unit)")
    ax.legend()
    ax.grid(alpha=0.2)

    ax2.scatter(freq / 1e6, signal - derivative_lorentz(x, *popt), s=11)
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Residual")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    output = Path(args.file).with_name(Path(args.file).stem + "_fit.png")
    fig.savefig(output, dpi=180)
    print(f"Graph saved:           {output}")
    plt.close()


if __name__ == "__main__":
    main()
