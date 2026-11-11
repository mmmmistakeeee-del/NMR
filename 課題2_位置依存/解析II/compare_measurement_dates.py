"""Compare retained historical fits with the newest fits at identical coordinates.

Writes diagnostic outputs only; never applies a correction to the position map.
Run from any directory with: python compare_measurement_dates.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_nmr_batch import read_data, fit_segment


def main():
    here = Path(__file__).resolve().parent
    old = pd.read_csv(here / 'results/fit_results.csv')
    new = pd.read_csv(here / 'results_with_additions/fit_results.csv')
    keys = ['x', 'z']
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError('Resolve multiple sweeps/duplicates before paired comparison.')
    pairs = old.merge(new, on=keys, suffixes=('_old', '_new'))
    pairs = pairs[pairs.date_old != pairs.date_new].sort_values(keys).copy()
    if pairs.empty:
        raise ValueError('No repeated coordinates from different dates.')
    pairs['delta_f0_Hz'] = pairs.f0_Hz_new - pairs.f0_Hz_old
    pairs['delta_formal_se_Hz'] = np.hypot(pairs.f0_se_Hz_new, pairs.f0_se_Hz_old)
    # Both dates use the same relative window, centered on their full-scan fit.
    # The 5 kHz half-window fits all eight scans; it is a sensitivity check,
    # not a replacement for the production fit or a systematic error estimate.
    for idx, pair in pairs.iterrows():
        for which in ('old', 'new'):
            folder = (here.parent / '実験II' if which == 'old' else
                      here.parent / '測定記録' / str(int(pair[f'date_{which}'])))
            raw, _, _ = read_data(folder / pair[f'filename_{which}'])
            if np.any(np.diff(raw[:, 0]) < 0):
                raise ValueError('Comparison currently requires upward-only scans.')
            center = pair[f'f0_Hz_{which}']
            if center - raw[:, 0].min() < 5000 or raw[:, 0].max() - center < 5000:
                raise ValueError('Insufficient scan coverage for +/-5 kHz check.')
            cropped = raw[np.abs(raw[:, 0] - center) <= 5000]
            fit, _ = fit_segment(cropped)
            pairs.loc[idx, f'window5k_f0_Hz_{which}'] = fit['f0_Hz']
            pairs.loc[idx, f'window5k_warning_{which}'] = fit['warning']
    pairs['window5k_delta_Hz'] = pairs.window5k_f0_Hz_new - pairs.window5k_f0_Hz_old
    out = here / 'results_date_comparison'
    out.mkdir(exist_ok=True)
    pairs.to_csv(out / 'paired_dates.csv', index=False)
    fig, (ax, dx) = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [f'x={r.x:g}, z={r.z:g}' for _, r in pairs.iterrows()]
    xx = np.arange(len(pairs))
    for which, label in [('old', 'Earlier measurement'), ('new', 'Later measurement')]:
        ax.errorbar(xx, (pairs[f'f0_Hz_{which}'] - 16e6) / 1000,
                    yerr=pairs[f'f0_se_Hz_{which}'] / 1000, marker='o', label=label)
    dx.errorbar(xx, pairs.delta_f0_Hz / 1000,
                yerr=pairs.delta_formal_se_Hz / 1000, fmt='o', label='Full scan fit')
    dx.scatter(xx, pairs.window5k_delta_Hz / 1000, marker='x', label='+/-5 kHz fit')
    dx.axhline(pairs.delta_f0_Hz.mean() / 1000, color='gray', ls='--', label='Mean full-scan shift')
    ax.set_ylabel('Resonance frequency - 16 MHz (kHz)')
    dx.set_ylabel('Later - earlier resonance frequency (kHz)')
    ax.set_title('Same coordinates, different dates')
    dx.set_title('Date-associated shift')
    for axes in (ax, dx):
        axes.set_xticks(xx, labels, rotation=20)
        axes.grid(alpha=.25)
        axes.legend(fontsize=8)
    fig.text(.5, .01, 'Error bars: formal fit SE only; correlated residuals and systematic errors are not included.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / 'paired_date_shift.png', dpi=170)
    plt.close(fig)
    print(pairs[['x', 'z', 'delta_f0_Hz', 'delta_formal_se_Hz', 'window5k_delta_Hz']].to_string(index=False))
    print('Mean shift / sample SD (Hz):', pairs.delta_f0_Hz.mean(), pairs.delta_f0_Hz.std())


if __name__ == '__main__':
    main()
