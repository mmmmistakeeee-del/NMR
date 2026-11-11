"""Batch derivative-Lorentz fits; see README.md and --help."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
NAME = re.compile(r'^(?P<date>\d{8}).*?_x(?P<x>[+-]?\d+(?:\.\d+)?)_z(?P<z>[+-]?\d+(?:\.\d+)?)(?=_|$)')
PARAMS = ['f0_Hz', 'hwhm_Hz', 'L0', 'baseline', 'baseline_slope_per_Hz']
SE = ['f0_se_Hz', 'hwhm_se_Hz', 'L0_se', 'baseline_se', 'baseline_slope_se_per_Hz']
FIELDS = ['filename','date','x','z','sweep','segment','sha256','n_raw','n_points','n_removed',
 'f_min_Hz','f_max_Hz','f_ref_Hz'] + PARAMS + SE + ['fwhm_Hz','fwhm_se_Hz','peak_to_peak_Hz',
 'peak_to_peak_se_Hz','residual_rms','noise_sigma_est','residual_lag1','signal_to_noise',
 'dof','jacobian_condition','n_starts','fit_status','warning','diagnostic_plot']


def derivative_lorentz(f, f0, gamma, L0, b0=0., b1=0., f_ref=0.):
    """Exactly the requested model, with frequency and absorption HWHM in Hz."""
    u = (np.asarray(f)-f0)/gamma
    return -2*(L0/gamma)*u/(1+u*u)**2+b0+b1*(np.asarray(f)-f_ref)


def read_data(path, unit='Hz'):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != 2:
        raise ValueError('Expected exactly two numeric columns: frequency and signal')
    good = np.all(np.isfinite(a), axis=1)
    b = a[good].copy()
    b[:,0] *= {'Hz':1., 'kHz':1e3, 'MHz':1e6}[unit]
    return b, len(a), int((~good).sum())


def split_sweeps(a):
    """Preserve acquisition order; disjoint monotonic legs, turn belongs to prior leg.
    Equal frequencies stay in the current leg. Short legs are not discarded.
    """
    if len(a)==0:
        return [('stationary',a)]
    starts, direction = [0], 0
    for i, step in enumerate(np.diff(a[:,0]),1):
        sign = int(np.sign(step))
        if sign and direction and sign!=direction:
            starts.append(i)
        if sign:
            direction=sign
    starts.append(len(a))
    result=[]
    for lo,hi in zip(starts[:-1],starts[1:]):
        part=a[lo:hi]
        d=part[-1,0]-part[0,0]
        result.append(('up' if d>0 else 'down' if d<0 else 'stationary',part))
    return result


def model_scaled(t,p,linear=True):
    c,logg,amp,offset=p[:4]
    u=(t-c)/np.exp(logg)
    return -2*amp*u/(1+u*u)**2+offset+(p[4]*t if linear else 0)


def fit_segment(a,linear=True):
    if len(a)<12 or len(np.unique(a[:,0]))<8:
        raise ValueError('Need >=12 points and >=8 distinct frequencies per sweep')
    f,y=a.T
    span=float(np.ptp(f))
    if span<=0 or np.ptp(y)==0:
        raise ValueError('Zero frequency or signal range')
    ref,yref,ys=float((f.min()+f.max())/2),float(np.median(y)),float(np.std(y))
    t,v=(f-ref)/span,(y-yref)/ys
    order=np.argsort(t)
    win=min(11,len(t) if len(t)%2 else len(t)-1)
    sm=savgol_filter(v[order],win,2)  # Initial guess ONLY; fit uses raw data.
    hi,lo=np.argmax(sm),np.argmin(sm)
    guess_c=float((t[order][hi]+t[order][lo])/2)
    guess_g=max(abs(t[order][hi]-t[order][lo])*np.sqrt(3)/2,.005)
    step=float(np.median(np.diff(np.unique(t))))
    gmin,gmax=max(step*.1,1e-7),2.
    lower=[-.5,np.log(gmin),-np.inf,-np.inf]+([-np.inf] if linear else [])
    upper=[.5,np.log(gmax),np.inf,np.inf]+([np.inf] if linear else [])
    # Variable projection: solve amplitude/background exactly at each (f0,Gamma).
    # Only two nonlinear parameters remain; covariance below still uses ALL parameters.
    def profile(q, coefficients=False):
        u=(t-q[0])/np.exp(q[1])
        cols=[-2*u/(1+u*u)**2,np.ones_like(t)]+([t] if linear else [])
        design=np.column_stack(cols)
        coeff=np.linalg.lstsq(design,v,rcond=None)[0]
        return coeff if coefficients else design@coeff-v
    candidates=[]
    for c in dict.fromkeys([guess_c,-.25,0.,.25]):
        for g in dict.fromkeys([float(np.clip(guess_g,gmin*1.1,gmax*.9)),.03,.12,.4]):
            opt=least_squares(profile,[c,np.log(g)],bounds=(lower[:2],upper[:2]),
                              max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            if opt.success and np.all(np.isfinite(opt.x)):
                candidates.append(opt)
    if not candidates:
        raise RuntimeError('No multistart fit converged')
    best=min(candidates,key=lambda r:np.dot(r.fun,r.fun))
    p=np.r_[best.x,profile(best.x,coefficients=True)]
    u=(t-p[0])/np.exp(p[1])
    kernel=(1-3*u*u)/(1+u*u)**3
    cols=[2*p[2]/np.exp(p[1])*kernel,2*p[2]*u*kernel,
          -2*u/(1+u*u)**2,np.ones_like(t)]+([t] if linear else [])
    full_jac=np.column_stack(cols)
    npar,dof=len(p),len(y)-len(p)
    gamma=float(span*np.exp(p[1]))
    values=np.array([ref+span*p[0],gamma,p[2]*ys*gamma,yref+ys*p[3],ys*p[4]/span if linear else 0.])
    residual=y-(yref+ys*model_scaled(t,p,linear))
    sigma=float(np.sqrt(np.dot(residual,residual)/dof))
    warn=[]
    _,singular,vt=np.linalg.svd(full_jac,full_matrices=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    cov=np.full((5,5),np.nan)
    if condition<1e10 and singular[-1]>np.finfo(float).eps*max(full_jac.shape)*singular[0]:
        cp=(vt.T/singular**2)@vt*np.dot(best.fun,best.fun)/dof
        # FULL Jacobian transformation, including L0=A*Gamma correlation.
        transform=np.zeros((5,npar))
        transform[0,0],transform[1,1]=span,gamma
        transform[2,1],transform[2,2],transform[3,3]=values[2],ys*gamma,ys
        if linear:
            transform[4,4]=ys/span
        cov=transform@cp@transform.T
    else:
        warn.append('ill_conditioned_covariance')
    errors=np.sqrt(np.maximum(np.diag(cov),0))
    if not linear:
        errors[4]=0.  # Fixed parameter.
    lag=float(np.corrcoef(residual[:-1],residual[1:])[0,1]) if np.std(residual)>0 else float('nan')
    snr=float(9/(8*np.sqrt(3))*abs(p[2])*ys/sigma) if sigma>0 else float('inf')
    if snr<5: warn.append('weak_signal')
    if abs(lag)>.3: warn.append('correlated_residuals')
    if values[0]-gamma/np.sqrt(3)<=f.min() or values[0]+gamma/np.sqrt(3)>=f.max():
        warn.append('derivative_extremum_outside_scan')
    if min(values[0]-f.min(),f.max()-values[0])<2*gamma:
        warn.append('short_baseline_range')
    if gamma<span*gmin*1.02 or gamma>span*gmax*.98 or np.any(best.active_mask):
        warn.append('parameter_at_bound')
    if gamma<np.median(np.diff(np.unique(f))): warn.append('width_below_sampling_step')
    if errors[1]>.5*gamma or errors[0]>gamma: warn.append('large_parameter_uncertainty')
    row=dict(zip(PARAMS,map(float,values)))
    row.update(dict(zip(SE,map(float,errors))))
    row.update(f_ref_Hz=ref,f_min_Hz=float(f.min()),f_max_Hz=float(f.max()),
        fwhm_Hz=2*gamma,fwhm_se_Hz=2*errors[1],peak_to_peak_Hz=2*gamma/np.sqrt(3),
        peak_to_peak_se_Hz=2*errors[1]/np.sqrt(3),residual_rms=float(np.sqrt(np.mean(residual**2))),
        noise_sigma_est=sigma,residual_lag1=lag,signal_to_noise=snr,dof=dof,
        jacobian_condition=condition,n_starts=len(candidates),fit_status='warning' if warn else 'ok',
        warning=';'.join(warn))
    return row,cov


def diagnostic(a,row,dest):
    f,y=a.T
    fig,(ax,rx)=plt.subplots(2,1,sharex=True,figsize=(9,6),gridspec_kw={'height_ratios':[3,1]})
    ref=row.get('f_ref_Hz',float(np.mean(f)))
    ax.scatter((f-ref)/1000,y,s=10,label='Data')
    if row['fit_status']!='failed':
        params=[row[k] for k in PARAMS]
        grid=np.linspace(f.min(),f.max(),1200)
        ax.plot((grid-ref)/1000,derivative_lorentz(grid,*params,f_ref=ref),color='C1',label='Fit')
        ax.axvline((row['f0_Hz']-ref)/1000,color='gray',ls='--')
        rx.scatter((f-ref)/1000,y-derivative_lorentz(f,*params,f_ref=ref),s=10)
        ax.text(.99,.97,f"f0 = {row['f0_Hz']/1e6:.8f} +/- {row['f0_se_Hz']/1e6:.8f} MHz\n"
            f"FWHM = {row['fwhm_Hz']:.1f} +/- {row['fwhm_se_Hz']:.1f} Hz (1 sigma)",
            ha='right',va='top',transform=ax.transAxes,fontsize=9)
    ax.set_title(f"{row['filename']} / {row['sweep']} #{row['segment']}",fontsize=10)
    ax.set_ylabel('Signal (original unit)')
    ax.legend(loc='lower right')
    rx.axhline(0,color='gray',lw=1)
    rx.set_ylabel('Residual')
    rx.set_xlabel(f'Frequency - {ref/1e6:.6f} MHz (kHz)')
    for axes in (ax,rx): axes.grid(alpha=.25)
    fig.text(.02,.015,row.get('warning','')[:180] or 'Fit status: ok',fontsize=8,color='darkred')
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(dest,dpi=150)
    plt.close(fig)


def select_position_scans(paths):
    """Choose the newest date, then the largest retake number at each (x, z)."""
    scan_keys = {}
    best = {}
    for path in paths:
        match = NAME.search(path.stem)
        if not match:
            continue  # Keep unrecognized names for the existing input error report.
        coordinate = (float(match['x']), float(match['z']))
        suffix = re.fullmatch(r'_([0-9]+)', path.stem[match.end():])
        revision = int(suffix[1]) if suffix else 1
        priority = (match['date'], revision)
        scan_keys[path] = (coordinate, priority)
        if coordinate not in best or priority > best[coordinate][0]:
            best[coordinate] = (priority, [path.name])
        elif priority == best[coordinate][0]:
            best[coordinate][1].append(path.name)

    selected, superseded = [], []
    for path in paths:
        if path in scan_keys:
            coordinate, priority = scan_keys[path]
            best_priority, best_names = best[coordinate]
            if priority < best_priority:
                superseded.append(dict(filename=path.name, x=coordinate[0], z=coordinate[1],
                                       date=priority[0], revision=priority[1],
                                       selected_date=best_priority[0], selected_revision=best_priority[1],
                                       selected_filenames=best_names))
                continue
        selected.append(path)
    return selected, superseded


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,nargs='?',default=HERE.parent/'実験II')
    parser.add_argument('--additional-input', type=Path, action='append', default=[],
                        help='Also fit .dat files in this directory. May be repeated.')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--unit',choices=['Hz','kHz','MHz'],default='Hz')
    parser.add_argument('--baseline',choices=['linear','constant'],default='linear')
    parser.add_argument('--fmin',type=float,help='Lower fit limit in Hz, after sweep separation')
    parser.add_argument('--fmax',type=float,help='Upper fit limit in Hz')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    if args.fmin is not None and args.fmax is not None and args.fmin>=args.fmax:
        parser.error('--fmin must be less than --fmax')
    files = []
    for directory in [args.input, *args.additional_input]:
        scans = sorted(directory.glob('*.dat'))
        if not scans:
            parser.error(f'No .dat files in {directory}')
        files.extend(scans)
    if len({path.name.casefold() for path in files}) != len(files):
        parser.error('Duplicate .dat filenames across input directories')
    input_file_count = len(files)
    files, superseded = select_position_scans(files)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not args.no_plots: (out/'diagnostics').mkdir(exist_ok=True)
    (out/'covariance').mkdir(exist_ok=True)
    rows,coords=[],[]
    for path in files:
        base={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        match=NAME.search(path.stem)
        if match:
            base.update(date=match['date'],x=float(match['x']),z=float(match['z']))
            coords.append((base['x'],base['z']))
        try:
            if not match: raise ValueError('Filename needs YYYYMMDD and _x<number>_z<number>')
            a,nraw,removed=read_data(path,args.unit)
            base.update(n_raw=nraw,n_removed=removed)
            segments=split_sweeps(a)
        except Exception as exc:
            rows.append(dict(base,fit_status='failed',warning=f'{type(exc).__name__}: {exc}'))
            continue
        for index,(direction,part) in enumerate(segments,1):
            row=dict(base,sweep=direction,segment=index)
            if args.fmin is not None: part=part[part[:,0]>=args.fmin]
            if args.fmax is not None: part=part[part[:,0]<=args.fmax]
            row['n_points']=len(part)
            key=f'{path.stem}__{index}_{direction}'
            try:
                result,cov=fit_segment(part,linear=args.baseline=='linear')
                row.update(result)
                extra=(['nonfinite_rows_removed'] if removed else [])+(['multiple_sweep_segments'] if len(segments)>1 else [])
                if extra:
                    row['warning']=';'.join(filter(None,[row['warning'],*extra]))
                    row['fit_status']='warning'
                with (out/'covariance'/f'{key}.csv').open('w',newline='',encoding='utf-8-sig') as handle:
                    writer=csv.writer(handle)
                    writer.writerow(['parameter',*PARAMS])
                    writer.writerows([[label,*line] for label,line in zip(PARAMS,cov)])
            except Exception as exc:
                row.update(fit_status='failed',warning=f'{type(exc).__name__}: {exc}')
            if not args.no_plots and len(part):
                dest=out/'diagnostics'/f'{key}.png'
                try:
                    diagnostic(part,row,dest)
                    row['diagnostic_plot']=str(dest.relative_to(out))
                except Exception as exc:
                    row['warning']=row.get('warning','')+f';plot_failed:{exc}'
                    if row['fit_status']=='ok': row['fit_status']='warning'
            rows.append(row)
            print(f"{path.name} #{index}: {row['fit_status']} {row.get('warning','')}",flush=True)
    with (out/'fit_results.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    duplicates=[{'x':k[0],'z':k[1],'files':n} for k,n in Counter(coords).items() if n>1]
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'input':str(args.input.resolve()),
        'unit':args.unit,'baseline':args.baseline,'fmin_Hz':args.fmin,'fmax_Hz':args.fmax,
        'files':len(files),'rows':len(rows),'statuses':dict(Counter(r['fit_status'] for r in rows)),
        'duplicate_coordinates':duplicates,'numpy_version':np.__version__,
        'error_definition':'1-sigma local least-squares covariance; iid equal-variance residuals; no systematics',
        'model':'-2*Gamma^2*L0*(f-f0)/(Gamma^2+(f-f0)^2)^2+b0+b1*(f-f_ref)'}
    if args.additional_input:
        metadata['additional_inputs'] = [str(directory.resolve()) for directory in args.additional_input]
    metadata['input_files'] = input_file_count
    metadata['superseded_files'] = superseded
    with (out/'run_summary.json').open('w',encoding='utf-8') as handle:
        json.dump(metadata,handle,indent=2,ensure_ascii=False)
    print(json.dumps(metadata['statuses']))
    print(f"CSV: {out/'fit_results.csv'}")
    if superseded: print(f'Superseded retakes skipped: {len(superseded)}')
    if duplicates: print(f'Repeated coordinates retained: {duplicates}')
    return 1 if any(r['fit_status']=='failed' for r in rows) else 0

if __name__=='__main__':
    raise SystemExit(main())
