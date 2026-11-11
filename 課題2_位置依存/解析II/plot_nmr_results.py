"""Plot fit_results.csv independently of raw data and fitting code."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
METRICS=[('f0_Hz','f0_se_Hz',1e-6,'Resonance frequency (MHz)'),
         ('fwhm_Hz','fwhm_se_Hz',1e-3,'Absorption FWHM (kHz)')]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path,nargs='?',default=HERE/'results'/'fit_results.csv')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--only-ok',action='store_true',help='Exclude warning fits')
    args=parser.parse_args(argv)
    with args.csv.open(encoding='utf-8-sig',newline='') as handle:
        raw=list(csv.DictReader(handle))
    rows=[]
    for source in raw:
        if source['fit_status'] not in (['ok'] if args.only_ok else ['ok','warning']): continue
        row=dict(source)
        for k in ['x','z','f0_Hz','f0_se_Hz','fwhm_Hz','fwhm_se_Hz']: row[k]=float(row[k])
        if row['x'] == 7 or row['z'] in (2, 7): continue
        if np.all(np.isfinite([row['x'],row['z'],row['f0_Hz'],row['fwhm_Hz']])): rows.append(row)
    if not rows: parser.error('No finite successful fits to plot')
    out=args.output or args.csv.parent/'plots'
    out.mkdir(parents=True,exist_ok=True)
    for value,err,scale,label in METRICS:
        for axis,fixed in [('x','z'),('z','x')]:
            groups=sorted(set(r[fixed] for r in rows))
            fig,ax=plt.subplots(figsize=(10,6))
            for idx,level in enumerate(groups):
                part=sorted([r for r in rows if r[fixed]==level],key=lambda r:r[axis])
                color=plt.get_cmap('tab20')(idx%20)
                valid_error=[r for r in part if np.isfinite(r[err]) and r[err]>=0]
                if valid_error:
                    ax.errorbar([r[axis] for r in valid_error],[r[value]*scale for r in valid_error],
                        yerr=[r[err]*scale for r in valid_error],fmt='none',ecolor=color,capsize=3)
                for status,marker in [('ok','o'),('warning','^')]:
                    points=[r for r in part if r['fit_status']==status]
                    if points:
                        ax.scatter([r[axis] for r in points],[r[value]*scale for r in points],
                                   color=color,marker=marker,s=30)
                ax.plot([],[],'o',color=color,label=f'{fixed}={level:g}')
            ax.set(xlabel=axis,ylabel=label,title='Error bars: statistical 1 sigma; triangles: warning')
            ax.ticklabel_format(axis='y',useOffset=False)
            ax.grid(alpha=.25)
            ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8)
            fig.tight_layout()
            fig.savefig(out/f'{value}_vs_{axis}.png',dpi=160)
            plt.close(fig)
        for key,title in [(value,label),(err,'Statistical 1 sigma: '+label)]:
            fig,ax=plt.subplots(figsize=(8,6))
            valid=[r for r in rows if np.isfinite(r[key])]
            if not valid:
                plt.close(fig)
                continue
            norm=plt.Normalize(min(r[key]*scale for r in valid),max(r[key]*scale for r in valid))
            for status,marker in [('ok','o'),('warning','^')]:
                part=[r for r in valid if r['fit_status']==status]
                if part:
                    ax.scatter([r['x'] for r in part],[r['z'] for r in part],
                        c=[r[key]*scale for r in part],cmap='viridis',norm=norm,
                        marker=marker,s=100,edgecolors='black',linewidths=.5)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label=title)
            counts={}
            for r in valid:
                xy=(r['x'],r['z'])
                counts[xy]=counts.get(xy,0)+1
            for (x,z),n in counts.items():
                if n>1: ax.annotate(f'n={n}',(x,z),xytext=(7,7),textcoords='offset points',fontsize=8)
            ax.set(xlabel='x',ylabel='z',title='Measured positions (no interpolation)')
            ax.set_aspect('equal',adjustable='box')
            ax.grid(alpha=.2)
            ax.margins(.1)
            ax.legend(handles=[Line2D([],[],color='black',marker='o',ls='',label='ok'),
                               Line2D([],[],color='black',marker='^',ls='',label='warning')],fontsize=8)
            fig.tight_layout()
            fig.savefig(out/f'{key}_xz.png',dpi=160)
            plt.close(fig)
    print(f'Plotted {len(rows)} rows ({len(raw)-len(rows)} excluded): {out.resolve()}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
