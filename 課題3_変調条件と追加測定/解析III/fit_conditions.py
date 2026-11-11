"""Fit the 2026-11-04 acquisition with the existing Task II baseline model.

Run with python -B fit_conditions.py. An existing output directory is refused.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NMR = HERE.parents[1]
INPUT_DIRS = (HERE.parent / '20261104', NMR / '課題2_位置依存' / '測定記録' / '20261104')
SETTINGS = json.loads((HERE / 'measurement_settings.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(NMR / '課題2_位置依存' / '解析II'))
import fit_nmr_batch as baseline

MODULATION_NAME = re.compile(r'_v([\d.]+)_f(\d+)(?:_(\d+))?$')


def select_inputs(paths):
    """Select one acquisition per modulation setting and the latest position scans."""
    tests, positions, modulation = [], [], {}
    for path in paths:
        if 'test' in path.stem.casefold():
            tests.append(path.name)
            continue
        match = MODULATION_NAME.search(path.stem)
        if match:
            key = (float(match[1]), int(match[2]))
            revision = int(match[3]) if match[3] else 1
            modulation.setdefault(key, []).append((revision, path))
        elif baseline.NAME.search(path.stem):
            positions.append(path)
        else:
            raise ValueError(f'Unknown condition name: {path.name}')

    selected, superseded_positions = baseline.select_position_scans(positions)
    superseded = [entry['filename'] for entry in superseded_positions]
    for key, runs in modulation.items():
        latest = max(revision for revision, _ in runs)
        winners = [path for revision, path in runs if revision == latest]
        if len(winners) != 1:
            raise ValueError(f'Multiple highest-numbered files for modulation setting {key}: '
                             + ', '.join(path.name for path in winners))
        selected.extend(winners)
        superseded.extend(path.name for revision, path in runs if revision < latest)
    return sorted(selected, key=lambda path: path.name), tests, sorted(superseded)


def conditions(path):
    match = MODULATION_NAME.search(path.stem)
    row = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               trial='test' in path.stem, voltage_V=np.nan, modulation_Hz=np.nan,
               x=np.nan, z=np.nan, voltage_series=False, frequency_series=False,
               position_series=False, delay_time_s=SETTINGS['delay_time_s'],
               sample_position=SETTINGS['fixed_sample_position'],
               measurement_settings='User confirmed central fixed position and 1 s delay; other settings unconfirmed')
    if match:
        row.update(voltage_V=float(match[1]), modulation_Hz=int(match[2]),
                   voltage_series=int(match[2]) == 280,
                   frequency_series=float(match[1]) == 1.3)
    else:
        match = re.search(r'_x([\d.]+)_z([\d.]+)', path.stem)
        if not match:
            raise ValueError(f'Unknown condition name: {path.name}')
        row.update(x=float(match[1]), z=float(match[2]), position_series=True,
                   sample_position=f'x={match[1]}, z={match[2]} (filename labels)')
    return row


def add_strength(row, covariance):
    gamma, amplitude = row['hwhm_Hz'], row['L0']
    strength = 9 / (4 * np.sqrt(3)) * abs(amplitude) / gamma
    gradient = np.zeros(5)
    gradient[1] = -strength / gamma
    gradient[2] = 9 / (4 * np.sqrt(3)) * np.sign(amplitude) / gamma
    variance = gradient @ covariance @ gradient
    row.update(signal_peak_to_peak=strength,
               signal_peak_to_peak_se=float(np.sqrt(max(variance, 0)))
               if np.isfinite(variance) else np.nan,
               peak_to_peak_se_Hz=2 * row['hwhm_se_Hz'] / np.sqrt(3))


def prediction(data, row):
    return baseline.derivative_lorentz(data[:, 0],
        *[row[key] for key in baseline.PARAMS], f_ref=row['f_ref_Hz'])


def overview(records, output):
    for start in range(0, len(records), 6):
        fig, axes = plt.subplots(6, 2, figsize=(13, 16))
        for pair in axes:
            for ax in pair:
                ax.set_visible(False)
        for pair, (data, row) in zip(axes, records[start:start + 6]):
            for ax in pair:
                ax.set_visible(True)
                ax.grid(alpha=.2)
            frequency = data[:, 0] / 1e6
            pair[0].plot(frequency, data[:, 1], '.', ms=3, label='Measured')
            pair[0].set_title(row['filename'], fontsize=9)
            if row['fit_status'] != 'failed':
                grid = np.column_stack([np.linspace(data[0, 0], data[-1, 0], 1200), np.zeros(1200)])
                pair[0].plot(grid[:, 0] / 1e6, prediction(grid, row), lw=1, label='Lorentz derivative')
                pair[1].plot(frequency, data[:, 1] - prediction(data, row), '.', ms=3)
                pair[1].set_title(f"Residual, lag-1 = {row['residual_lag1']:.2f}", fontsize=9)
            pair[0].set_ylabel('Recorded signal')
            pair[1].axhline(0, color='gray', lw=.6)
            for ax in pair:
                ax.set_xlabel('RF frequency (MHz)')
                ax.ticklabel_format(useOffset=False)
                ax.xaxis.set_major_locator(MaxNLocator(5))
        fig.tight_layout()
        fig.savefig(output / f'overview_{start // 6 + 1:02d}.png', dpi=130)
        plt.close(fig)


def trend_plots(table, output):
    metrics = [('f0_Hz', 'f0_se_Hz', 1e-6, 'Resonance frequency (MHz)'),
               ('fwhm_Hz', 'fwhm_se_Hz', 1e-3, 'Lorentz-model absorption FWHM (kHz)'),
               ('peak_to_peak_Hz', 'peak_to_peak_se_Hz', 1e-3, 'Signal peak separation (kHz)'),
               ('signal_peak_to_peak', 'signal_peak_to_peak_se', 1., 'Signal peak-to-peak (recorded units)')]
    for series, column, label in [('voltage', 'voltage_V', 'Modulation voltage (V, name label)'),
                                   ('frequency', 'modulation_Hz', 'Modulation frequency (Hz, name label)'),
                                   ('position', 'x', 'x (coordinate label), z = 4.0')]:
        part = table[table[series + '_series'] & (table.fit_status != 'failed')]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (value, error, scale, ylabel) in zip(axes.flat, metrics):
            for status, marker, color in [('ok', 'o', '#16735a'), ('warning', '^', '#b66b18')]:
                subset = part[part.fit_status == status]
                if len(subset):
                    ax.errorbar(subset[column], subset[value] * scale, yerr=subset[error] * scale,
                                fmt=marker, color=color, capsize=3, ms=5, label=status)
            ax.set(xlabel=label, ylabel=ylabel)
            ax.ticklabel_format(axis='y', useOffset=False)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f'{series.capitalize()} dependence | highest numbered acquisition')
        fig.text(.5, .01, 'Bars: local iid 1-sigma only. Instrument settings unconfirmed.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .95))
        fig.savefig(output / f'{series}_dependence.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    for directory in INPUT_DIRS:
        if not any(directory.glob('*.dat')):
            parser.error(f'No .dat files in {directory}')
    files = sorted((path for directory in INPUT_DIRS for path in directory.glob('*.dat')),
                   key=lambda path: path.name)
    if len({path.name for path in files}) != len(files):
        parser.error('Duplicate input filenames across the two measurement directories')
    input_count = len(files)
    try:
        files, tests, superseded = select_inputs(files)
    except ValueError as error:
        parser.error(str(error))
    if not files:
        parser.error('No measurements remained after input selection')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['diagnostics', 'covariance', 'plots']:
        (output / name).mkdir()
    rows, manifest, windows, records = [], [], [], []
    for path in files:
        metadata = conditions(path)
        data, raw_count, removed = baseline.read_data(path)
        manifest.append(dict(metadata, n_raw=raw_count, n_removed=removed,
                             f_min_Hz=data[:, 0].min(), f_max_Hz=data[:, 0].max(),
                             median_step_Hz=np.median(np.diff(data[:, 0]))))
        for segment, (direction, points) in enumerate(baseline.split_sweeps(data), 1):
            row = dict(metadata, segment=segment, sweep=direction, n_raw=raw_count,
                       n_removed=removed, n_points=len(points))
            try:
                fit, covariance = baseline.fit_segment(points)
                row.update(fit)
                add_strength(row, covariance)
                if removed:
                    row['warning'] = ';'.join(filter(None, [row['warning'], 'nonfinite_rows_removed']))
                    row['fit_status'] = 'warning'
                row['review_status'] = 'settings_and_fit_review'
                pd.DataFrame(covariance, index=baseline.PARAMS, columns=baseline.PARAMS).to_csv(
                    output / 'covariance' / f'{path.stem}__{segment}.csv')
                for radius in [10000., 20000.]:
                    mask = abs(points[:, 0] - row['f0_Hz']) <= radius
                    subset = points[mask]
                    check = dict(filename=path.name, segment=segment, radius_Hz=radius,
                                 n_points=len(subset), fit_status='failed')
                    try:
                        cropped, cov = baseline.fit_segment(subset)
                        add_strength(cropped, cov)
                        check.update(cropped)
                        check.update(f0_change_Hz=cropped['f0_Hz'] - row['f0_Hz'],
                                     fwhm_change_percent=100 * (cropped['fwhm_Hz'] / row['fwhm_Hz'] - 1),
                                     signal_change_percent=100 * (cropped['signal_peak_to_peak'] / row['signal_peak_to_peak'] - 1))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        check['warning'] = str(error)
                    windows.append(check)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row.update(fit_status='failed', warning=str(error))
            rows.append(row)
            records.append((points, row))
            baseline.diagnostic(points, row, output / 'diagnostics' / f'{path.stem}__{segment}.png')
            print(path.name, row['fit_status'], row.get('warning', ''), flush=True)
    table = pd.DataFrame(rows)
    pd.DataFrame(manifest).to_csv(output / 'measurement_manifest.csv', index=False, encoding='utf-8-sig')
    table.to_csv(output / 'fit_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(windows).to_csv(output / 'window_sensitivity.csv', index=False, encoding='utf-8-sig')
    overview(records, output / 'plots')
    trend_plots(table, output / 'plots')
    for path, item in zip(files, manifest, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    summary = dict(input_files=input_count, files=len(files), rows=len(rows),
                   excluded_test_files=tests, superseded_files=superseded,
                   statuses=table.fit_status.value_counts().to_dict(),
                   raw_data_sha256_verified=True, model='derivative Lorentz + linear background',
                   settings=SETTINGS,
                   signal_strength='Peak-to-peak of baseline-subtracted fitted signal',
                   uncertainty='Local iid least-squares covariance; parameter covariance included for strength',
                   software={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                             [('fit_conditions.py', Path(__file__)), ('fit_nmr_batch.py', Path(baseline.__file__))]})
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['statuses']), flush=True)


if __name__ == '__main__':
    main()
