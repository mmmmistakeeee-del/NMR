"""Compare representative scans using existing finite-modulation Voigt shapes.

Independent fits with free amplitudes and backgrounds. Quasi-static response.
Profile curves describe width tradeoffs; they are not confidence intervals.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.special import voigt_profile

import fit_conditions as base
sys.path.insert(0, str(base.NMR / '課題1_共鳴信号' / '解析I'))
from nmr_joint_noise import modulated_voigt

HERE = Path(__file__).resolve().parent
REPRESENTATIVES = [(0.5, 280), (1.3, 280), (4.5, 280), (1.3, 20), (1.3, 400)]


def predict(x, q, beta, nodes=256):
    return beta[0] * modulated_voigt(x - q[0], *q[1:], nodes=nodes) + beta[1] + beta[2] * x


def fit_shape(x, y, starts, upper_width, fixed=None, nodes=128):
    fixed = fixed or {}
    free = [i for i in range(4) if i not in fixed]
    lower = np.array([x.min(), .02, 0., 0.])
    upper = np.array([x.max(), upper_width, upper_width, upper_width])
    scale = np.std(y)

    def expand(values):
        q = np.zeros(4)
        q[free] = values
        for key, value in fixed.items():
            q[key] = value
        return q

    def solve(q):
        matrix = np.column_stack([modulated_voigt(x - q[0], *q[1:], nodes=nodes), np.ones(len(x)), x])
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        return matrix @ beta, beta

    candidates = []
    for start in starts:
        guess = np.clip(np.asarray(start)[free], lower[free] + 1e-7, upper[free] - 1e-7)
        result = least_squares(lambda values: (solve(expand(values))[0] - y) / scale,
                               guess, bounds=(lower[free], upper[free]), max_nfev=700,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if result.success:
            q = expand(result.x)
            fitted, beta = solve(q)
            candidates.append((np.sum((y - fitted)**2), q, beta, fitted))
    if not candidates:
        raise RuntimeError('Finite-modulation optimization did not converge')
    sse, q, beta, fitted = min(candidates, key=lambda item: item[0])
    fine = predict(x, q, beta, nodes=2 * nodes)
    error = float(np.max(abs(fine - fitted)))
    tolerance = 1e-4 * max(np.sqrt(sse / len(x)), 1e-10)
    if error > tolerance:
        if nodes >= 1024:
            raise RuntimeError(f'Quadrature convergence failed: {error}')
        return fit_shape(x, y, [q, *starts], upper_width, fixed, nodes=2 * nodes)
    return dict(sse=float(sse), q=q, beta=beta, prediction=fitted,
                quadrature_error=error, nodes=nodes,
                at_bound=any(q[i] - lower[i] < 1e-4 or upper[i] - q[i] < 1e-4 for i in free))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    table = pd.read_csv(args.results / 'fit_results.csv')
    selected = []
    for voltage, frequency in REPRESENTATIVES:
        matches = table[(table.voltage_V == voltage) & (table.modulation_Hz == frequency)]
        if len(matches) != 1 or matches.iloc[0].fit_status == 'failed':
            parser.error(f'Expected one successful fit at {voltage} V, {frequency} Hz')
        selected.append(matches.iloc[0])
    output = args.results / 'model_comparison'
    output.mkdir(exist_ok=False)
    rows, profiles, details = [], [], {}
    for row in selected:
        name = row.filename
        data = np.loadtxt(HERE.parent / '20261104' / name)
        ref = row.f0_Hz
        x, y = (data[:, 0] - ref) / 1000, data[:, 1]
        width = row.hwhm_Hz / 1000
        upper_width = max(10., 4 * width)
        starts = [[0, width, 0, 0], [0, .6 * width, .5 * width, .5 * width],
                  [0, .3 * width, .1 * width, width], [0, .3 * width, width, .1 * width]]
        fits = {}
        for kind, fixed in [('finite_lorentz', {2: 0.}), ('voigt_derivative', {3: 0.}), ('finite_voigt', {})]:
            extra = [fit['q'] for fit in fits.values()]
            fit = fit_shape(x, y, starts + extra, upper_width, fixed)
            fits[kind] = fit
            q, beta = fit['q'], fit['beta']
            half = voigt_profile(0., q[2], q[1]) / 2
            half_width = brentq(lambda offset: voigt_profile(offset, q[2], q[1]) - half, 0, 20 * upper_width)
            residual = y - fit['prediction']
            rows.append(dict(filename=name, model=kind, f0_Hz=ref + 1000 * q[0],
                             gamma_kHz=q[1], sigma_kHz=q[2], modulation_peak_kHz=q[3],
                             unmodulated_absorption_fwhm_kHz=2 * half_width,
                             residual_rms=np.sqrt(fit['sse'] / len(x)),
                             residual_lag1=np.corrcoef(residual[:-1], residual[1:])[0, 1],
                             rms_ratio_to_lorentz=np.sqrt(fit['sse'] / len(x)) / row.residual_rms,
                             at_bound=fit['at_bound'], quadrature_nodes=fit['nodes'],
                             quadrature_max_error=fit['quadrature_error']))
            print(name, kind, 'widths', q[1:], 'RMS', rows[-1]['residual_rms'], flush=True)
        full = fits['finite_voigt']
        if row.voltage_series:
            amplitudes = np.unique(np.r_[np.linspace(0, max(3., 1.5 * width), 9), full['q'][3]])
            previous = full['q']
            for amplitude in amplitudes:
                fit = fit_shape(x, y, [previous, full['q'], *starts[:2]], upper_width, fixed={3: amplitude})
                previous = fit['q']
                profiles.append(dict(filename=name, modulation_peak_kHz=amplitude,
                                     gamma_kHz=fit['q'][1], sigma_kHz=fit['q'][2],
                                     sse=fit['sse'], sse_ratio=fit['sse'] / full['sse'],
                                     quadrature_max_error=fit['quadrature_error']))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        grid = np.linspace(x.min(), x.max(), 1800)
        baseline_prediction = base.prediction(data, row)
        baseline_grid = np.column_stack([ref + 1000 * grid, np.zeros(len(grid))])
        baseline_values = base.prediction(baseline_grid, row)
        for ax in axes[0]:
            ax.plot(x, y, '.', color='black', ms=3, label='Measured')
            ax.plot(grid, baseline_values, '--', color='gray', lw=1, label='Lorentz derivative')
        for ax in axes[1]:
            ax.plot(x, y - baseline_prediction, '.', color='gray', ms=3, label='Lorentz residual')
        for kind, fit in fits.items():
            for ax in axes[0]:
                ax.plot(grid, predict(grid, fit['q'], fit['beta'], nodes=2 * fit['nodes']), lw=1, label=kind)
            for ax in axes[1]:
                ax.plot(x, y - fit['prediction'], lw=.8, label=kind)
        radius = max(6., 3 * width)
        axes[0, 1].set_xlim(-radius, radius)
        axes[1, 1].set_xlim(-radius, radius)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.set_xlabel(f'RF offset from {ref / 1e6:.6f} MHz (kHz)')
        axes[0, 0].set_ylabel('Recorded signal')
        axes[1, 0].set_ylabel('Residual')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(name + ' | independent shapes, linear background')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(output / (Path(name).stem + '.png'), dpi=150)
        plt.close(fig)
        details[name] = {kind: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                for key, value in fit.items() if key != 'prediction'} for kind, fit in fits.items()}
        pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(profiles).to_csv(output / 'width_profiles.csv', index=False, encoding='utf-8-sig')
        (output / 'parameters.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    frame = pd.DataFrame(profiles)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for pair, (name, group) in zip(axes, frame.groupby('filename', sort=False)):
        pair[0].plot(group.modulation_peak_kHz, group.sse_ratio, 'o-')
        pair[0].set_ylabel('SSE / best free-fit SSE')
        pair[1].plot(group.modulation_peak_kHz, group.gamma_kHz, 'o-', label='Lorentz HWHM')
        pair[1].plot(group.modulation_peak_kHz, group.sigma_kHz, 's-', label='Gaussian sigma')
        pair[1].set_ylabel('Width (kHz)')
        pair[1].legend(fontsize=8)
        for ax in pair:
            ax.set_title(name, fontsize=9)
            ax.set_xlabel('Fixed modulation peak excursion (kHz)')
            ax.grid(alpha=.2)
    fig.suptitle('Width tradeoffs | profile diagnostic, no confidence threshold assigned')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'width_tradeoffs.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
