"""測定データから、平均波形・幅の交換関係・残差ノイズを解析する。

平均波形とノイズまで:
    python -B nmr_joint_noise.py 20261020_KITtest01.dat

標準不確かさまで（模擬測定300回。前回は約10分）:
    python -B nmr_joint_noise.py 20261020_KITtest01.dat --bootstrap 300

保存済みの結果を読むだけなら nmr_uncertainty.py --summarize-only を使う。
全体の案内は同じフォルダの README.md を参照。

平均波形 = 信号の倍率 × 有限変調Voigt波形 + 直線的な背景
  Voigt: ローレンツ吸収に、共鳴周波数のガウス分布を重ねたもの。
  有限変調: 単なる微分近似でなく、正弦波変調の基本波成分を計算する。

mainの流れ:
  1. 読み込みと単位換算 → 2. 平均波形のフィット → 3. 幅の交換関係
  → 4. 平均を固定したノイズ比較 → 5. 結果保存・作図 → 6. 任意で不確かさ

内部の周波数差と幅はkHz。q=[中心の差, gamma, sigma, a]、
beta=[信号の倍率, 背景切片, 背景の傾き]。fとrefはHz。
平均波形は非加重最小二乗で推定し、ノイズはその残差に後から当てはめる。
相関は測定点の順序で評価する。秒単位の時定数や固有T2は推定しない。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares, minimize
from scipy.special import voigt_profile

from nmr_physical_models import modulated_lorentz, voigt_derivative


def modulated_voigt(x, gamma, sigma, excursion, nodes=48):
    """Voigt吸収を磁場変調したときに観測する波形を返す。

    x: 中心からの周波数差 [kHz]
    gamma: ローレンツ吸収の半値半幅 [kHz]
    sigma: 共鳴周波数のガウス分布の標準偏差 [kHz]
    excursion (=a): 変調の片側周波数振幅 [kHz]
    nodes: 数値積分の点数（物理パラメータではない）

    出力は基本波/a。a→0ならVoigt吸収の微分になる。
    この正規化の倍率は、後で自由パラメータAに含める。
    したがってa=0の解は「微小変調近似」を意味する。

    部分積分で S1/a = integral[-1,1] (2/pi)*sqrt(1-t*t)*V'(x+a*t) dt
    と書き換え、第2種Gauss-Chebyshev積分で数値計算する。
    変調中も磁化が追従する、準静的な応答を仮定している。
    """
    if sigma < 1e-4*gamma:
        return modulated_lorentz(x, gamma, excursion)
    if excursion < 1e-8:
        return voigt_derivative(x, gamma, sigma)
    theta = np.arange(1, nodes+1)*np.pi/(nodes+1)
    weights = 2*np.sin(theta)**2/(nodes+1)
    samples = np.asarray(x)[..., None] + excursion*np.cos(theta)
    return voigt_derivative(samples, gamma, sigma)@weights


def fit_mean(x, y, starts, fixed_a=None, fixed_sigma=None, nodes=48):
    """平均波形を非加重最小二乗でフィットする。

    x: 基準周波数からの差 [kHz]、y: 測定信号 [元データ単位]。
    q=[中心の差c, ローレンツ半値半幅gamma, ガウス分布幅sigma, 変調片振幅a]。
    beta=[信号の倍率A, 背景切片b0, 背景の傾きb1]。
    モデルは y = A * modulated_voigt(x-c, gamma, sigma, a) + b0 + b1*x。

    qを少しずつ変えて形を探し、各qでA/b0/b1を線形最小二乗で決める。
    7個を一度に非線形最適化するより、探索を小さくできる。
    fixed_a / fixed_sigmaを指定すると、その幅を固定して他を調整する。
    戻り値は (二乗残差和, q, beta, 各測定点のモデル値)。
    """
    fixed_parameters = {
        index: value
        for index, value in [(2, fixed_sigma), (3, fixed_a)]
        if value is not None
    }
    free_indices = [index for index in range(4) if index not in fixed_parameters]
    lower = np.array([max(float(x.min()), -5.), .02, 0., 0.])
    upper = np.array([min(float(x.max()), 5.), 5., 3., 3.])

    def expand(free_values):
        """自由な変数だけの配列に固定値を戻し、4要素のqを作る。"""
        shape_parameters = np.zeros(4)
        shape_parameters[free_indices] = free_values
        for index, value in fixed_parameters.items():
            shape_parameters[index] = value
        return shape_parameters

    def solve(shape_parameters):
        """波形の形を固定した状態で、倍率と背景を最適化する。"""
        center = shape_parameters[0]
        line_shape = modulated_voigt(x-center, *shape_parameters[1:], nodes=nodes)
        design_matrix = np.column_stack([line_shape, np.ones(len(x)), x])
        linear_parameters = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
        prediction = design_matrix @ linear_parameters
        return prediction, linear_parameters

    results = []
    for start in starts:
        start = np.clip(np.asarray(start)[free_indices], lower[free_indices]+1e-9, upper[free_indices]-1e-9)
        result = least_squares(lambda v: solve(expand(v))[0]-y, start,
                               bounds=(lower[free_indices], upper[free_indices]), max_nfev=450,
                               ftol=2e-9, xtol=2e-9, gtol=2e-9)
        if result.success:
            q = expand(result.x)
            pred, beta = solve(q)
            results.append((float(np.sum((y-pred)**2)), q, beta, pred))
    if not results:
        raise RuntimeError("Mean fit did not converge")
    return min(results, key=lambda item: item[0])


def innovations(residual, white_variance, hf, hg, rho):
    """過去の残差から次の点を予測し、予測できなかった部分を取り出す。

    residual_i = hf_i * U_i + hg_i * G_i + e_i,
    hf = 周波数揺らぎの大きさ × 平均信号の傾き
    hg = 相対振幅揺らぎの大きさ × 平均信号
    U/Gは分散1で共通の隣接相関rhoを持つ。eは独立な電圧雑音。

    カルマンフィルターで予測誤差と予測分散を逐次計算する。
    予測誤差 / sqrt(予測分散) が標準化イノベーション。
    戻り値: (負の対数尤度, 標準化イノベーション, 予測分散)。
    負の対数尤度には分散の対数項も含め、大きな分散で逃げることを防ぐ。
    """
    uf = ug = 0.
    pff = pgg = 1.
    pfg = 0.
    rho2 = rho*rho
    out = np.empty(len(residual))
    variance = np.empty(len(residual))
    for i in range(len(residual)):
        a, b = hf[i], hg[i]
        pf = pff*a+pfg*b
        pg = pfg*a+pgg*b
        v = white_variance[i]+a*pf+b*pg
        error = residual[i]-a*uf-b*ug
        out[i] = error/np.sqrt(v)
        variance[i] = v
        uf = rho*(uf+pf*error/v)
        ug = rho*(ug+pg*error/v)
        pff = rho2*(pff-pf*pf/v)+(1-rho2)
        pgg = rho2*(pgg-pg*pg/v)+(1-rho2)
        pfg = rho2*(pfg-pf*pg/v)
    nll = .5*np.sum(np.log(2*np.pi*variance)+out*out)
    return float(nll), out, variance


def fit_noise(name, residual, xscaled, signal, derivative):
    """平均波形を固定した残差に、指定したノイズモデルを当てはめる。

    log_sigma: 背景電圧雑音の標準偏差の対数
    trend: その対数標準偏差の周波数依存係数（xscaledは標準化した周波数）
    jitter: 周波数揺らぎの標準偏差 [kHz]
    gain: 相対振幅揺らぎの標準偏差 [割合。0.08なら8%]
    rho: 相関成分の隣接点間相関

    平均波形とノイズを一度に尤度最大化しているわけではない。
    戻り値: (ノイズのパラメータ辞書, 負の対数尤度, 標準化予測誤差,
              予測分散, 自由なノイズパラメータ数)。
    """
    specs = {
        "constant_white": [],
        "varying_white": ["trend"],
        "white_plus_jitter": ["trend", "jitter"],
        "white_plus_gain": ["trend", "gain"],
        "white_plus_both": ["trend", "jitter", "gain"],
        "white_plus_correlated_jitter": ["trend", "jitter", "rho"],
        "white_plus_correlated_gain": ["trend", "gain", "rho"],
        "white_plus_correlated_both": ["trend", "jitter", "gain", "rho"],
    }
    active = ["log_sigma"]+specs[name]
    limits = {"log_sigma": (np.log(.005), np.log(.3)), "trend": (-1., 1.),
              "jitter": (0., .5), "gain": (0., .5), "rho": (-.95, .98)}

    def unpack(values):
        p = dict(trend=0., jitter=0., gain=0., rho=0.)
        p.update(zip(active, values))
        return p

    def evaluate(values):
        p = unpack(values)
        white = np.exp(2*(p["log_sigma"]+p["trend"]*xscaled))
        return innovations(residual, white, p["jitter"]*derivative,
                           p["gain"]*signal, p["rho"])

    candidates = []
    for initial_rho in ([0., .65, .9] if "rho" in active else [0.]):
        initial = dict(log_sigma=np.log(.05), trend=.1, jitter=.03, gain=.04, rho=initial_rho)
        result = minimize(lambda v: evaluate(v)[0], [initial[k] for k in active],
                          bounds=[limits[k] for k in active], method="L-BFGS-B",
                          options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-6})
        if result.success:
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"Noise fit did not converge: {name}")
    result = min(candidates, key=lambda item: item.fun)
    p = unpack(result.x)
    nll, z, variance = evaluate(result.x)
    return p, nll, z, variance, len(active)


def acf(y, lag=1):
    return float(np.corrcoef(y[:-lag], y[lag:])[0,1])


def validate_math():
    # Independent first-harmonic integral; also covers strong modulation.
    for gamma, sigma, a in [(.5, 1., 1.2), (.2, .4, 2.), (1., .1, .5)]:
        for x in [-2., -.3, .8]:
            exact = quad(lambda t: voigt_profile(x+a*np.cos(t), sigma, gamma)*np.cos(t),
                         0, 2*np.pi, epsabs=1e-10)[0]/(np.pi*a)
            actual = modulated_voigt(np.array([x]), gamma, sigma, a, nodes=128)[0]
            assert np.isclose(actual, exact, rtol=1e-7, atol=1e-9)
    # Compare Kalman innovations and likelihood against full dense covariance.
    rng = np.random.default_rng(20261027)
    n=25
    residual = rng.normal(size=n)
    white = rng.uniform(.2, .5, n)
    hf, hg = rng.normal(size=(2,n))
    rho=.72
    lag=abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    c = np.diag(white)+(np.outer(hf,hf)+np.outer(hg,hg))*rho**lag
    chol = cholesky(c, lower=True)
    expected_z = solve_triangular(chol, residual, lower=True)
    expected_nll = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+expected_z@expected_z)
    nll,z,_=innovations(residual,white,hf,hg,rho)
    assert np.allclose(z, expected_z, atol=1e-10)
    assert abs(nll-expected_nll)<1e-9
    print("変調の数値積分とカルマン尤度を、独立の計算で確認しました。", flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default=str(Path(__file__).with_name("20261020_KITtest01.dat")),
                        help="周波数[Hz]・信号の2列ファイル")
    parser.add_argument("--validate-only", action="store_true", help="数値式の検証だけを実行する")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="不確かさの模擬測定回数（省略時は不確かさを計算しない）")
    parser.add_argument("--workers", type=int, default=4, help="並列で計算する数（既定: 4）")
    parser.add_argument("--seed", type=int, default=20261028, help="乱数の再現用番号")
    args=parser.parse_args()
    validate_math()
    if args.validate_only:
        return
    # 1. データを読み、数値が大きくなりすぎないよう周波数差[kHz]に換算する。
    src=Path(args.file)
    data=np.loadtxt(src)
    f,y=data[:,0],data[:,1]
    if not np.isfinite(data).all() or np.any(np.diff(f)<=0):
        raise ValueError("Finite, strictly increasing frequency data required; order is preserved.")
    ref=float((f[np.argmax(y)]+f[np.argmin(y)])/2)
    x=(f-ref)/1000
    core=abs(x)<5
    prefix=src.with_name(src.stem+"_joint_noise")
    # 2. 平均波形を推定する。初期値を複数使い、局所的な解への依存を減らす。
    starts=[[.02,.48,1.03,0.], [.02,.6,.8,1.], [.02,1.07,0.,1.33],
            [.02,.3,1.,.6], [.02,.8,.5,1.3]]
    fit=fit_mean(x,y,starts)
    # Refine and check all selected final curves at 128 nodes.
    fit=fit_mean(x,y,[fit[1],starts[0],starts[2]],nodes=128)
    sse,q,beta,pred=fit
    print(f"平均波形の初期フィット: 二乗残差和={sse:.7f}; "
          f"gamma={q[1]:.4f}, sigma={q[2]:.4f}, 変調片振幅={q[3]:.4f} kHz", flush=True)

    # 3. 変調幅aを固定して他の幅を再調整し、幅の交換関係を調べる。
    profiles=[]
    previous=q.copy()
    for a in np.linspace(0,2.,21):
        guess=np.array([.02,.5,max(.05,np.sqrt(max(1.07-a*a/4,.01))),a])
        candidate=fit_mean(x,y,[previous,guess],fixed_a=float(a))
        # Include the exact zero-Gaussian-width boundary separately.
        boundary=fit_mean(x,y,[[.02,1.,0.,a]],fixed_a=float(a),fixed_sigma=0.)
        candidate=min([candidate,boundary],key=lambda v:v[0])
        psse,pq,pb,pp=candidate
        check=pb[0]*modulated_voigt(x-pq[0],*pq[1:],nodes=128)+pb[1]+pb[2]*x
        if np.max(abs(check-pp))>1e-6:
            candidate=fit_mean(x,y,[pq],fixed_a=float(a),nodes=128)
            psse,pq,pb,pp=candidate
        if psse<sse:
            sse,q,beta,pred=candidate
        previous=pq
        profiles.append(dict(modulation_peak_kHz=a, gamma_kHz=pq[1],sigma_kHz=pq[2],
                             gamma_at_lower_bound=bool(pq[1]<.02001),
                             f0_MHz=(ref+1000*pq[0])/1e6,SSE=psse,
                             RMS=np.sqrt(psse/len(y)),core_RMS=np.sqrt(np.mean((y[core]-pp[core])**2))))
        print(f"幅の交換関係: 変調片振幅a={a:.1f} kHzに固定; "
              f"gamma={pq[1]:.4f}, sigma={pq[2]:.4f} kHz; 二乗残差和={psse:.7f}", flush=True)
    profile=pd.DataFrame(profiles)
    profile["delta_SSE"]=profile.SSE-sse
    profile["iid_reference_delta_deviance"]=len(y)*np.log(profile.SSE/sse)
    profile.to_csv(str(prefix)+"_profile.csv",index=False)

    # 4. 推定した平均波形を固定し、その残差に8種類のノイズを比較する。
    residual=y-pred
    signal=pred-beta[1]-beta[2]*x
    h=1e-4
    derivative=beta[0]*(modulated_voigt(x+h-q[0],*q[1:],nodes=128)
                        -modulated_voigt(x-h-q[0],*q[1:],nodes=128))/(2*h)
    xmean=float(x.mean()); xstd=float(x.std())
    xscaled=(x-xmean)/xstd
    names=["constant_white","varying_white","white_plus_jitter","white_plus_gain",
           "white_plus_both","white_plus_correlated_jitter","white_plus_correlated_gain",
           "white_plus_correlated_both"]
    rows=[]; noises={}
    for name in names:
        p,nll,z,var,k=fit_noise(name,residual,xscaled,signal,derivative)
        white=np.exp(p["log_sigma"]+p["trend"]*xscaled)
        row=dict(model=name,noise_parameters=k,NLL=nll,conditional_score=2*nll+2*k,
                 white_SD_at_ref=np.exp(p["log_sigma"]-p["trend"]*xmean/xstd),
                 jitter_SD_Hz=1000*p["jitter"],gain_SD_percent=100*p["gain"],
                 latent_rho_per_observation=p["rho"],innovation_RMS=np.sqrt(np.mean(z*z)),
                 innovation_core_RMS=np.sqrt(np.mean(z[core]**2)),
                 innovation_lag1=acf(z),innovation_core_lag1=acf(z[core]),
                 innovation_core_mean=z[core].mean())
        rows.append(row)
        noises[name]=(p,z,var,white)
        print(f"ノイズ候補: {name}; 比較スコア={row['conditional_score']:.2f}; "
              f"共鳴付近の標準化予測誤差の隣接相関={row['innovation_core_lag1']:.3f}", flush=True)
    noise_table=pd.DataFrame(rows)
    noise_table["delta_conditional_score"]=noise_table.conditional_score-noise_table.conditional_score.min()
    noise_table.to_csv(str(prefix)+"_noise.csv",index=False)
    best_name=str(noise_table.loc[noise_table.conditional_score.idxmin(),"model"])
    p,z,var,white=noises[best_name]
    pointwise=pd.DataFrame(dict(frequency_Hz=f,signal=y,mean_fit=pred,residual=residual,
                               fitted_resonance=signal,signal_derivative_per_kHz=derivative,
                               white_SD=white,innovation=z,innovation_SD=np.sqrt(var)))
    pointwise.to_csv(str(prefix)+"_residuals.csv",index=False)

    # 5. フィット範囲を変えた感度も保存し、数表と図を出力する。
    window_rows=[]
    for radius in [5,10,15,30]:
        mask=abs(x)<radius
        fsse,fq,fb,fp=fit_mean(x[mask],y[mask],[q,starts[0],starts[2]],nodes=128)
        window_rows.append(dict(half_window_kHz=radius,n=int(sum(mask)),gamma_kHz=fq[1],
                                sigma_kHz=fq[2],modulation_peak_kHz=fq[3],
                                f0_MHz=(ref+1000*fq[0])/1e6,RMS=np.sqrt(fsse/sum(mask))))
        print(f"フィット範囲の確認: ±{radius} kHz, {sum(mask)}点; "
              f"共鳴周波数={window_rows[-1]['f0_MHz']:.8f} MHz", flush=True)
    pd.DataFrame(window_rows).to_csv(str(prefix)+"_windows.csv",index=False)
    validation_error=float(np.max(abs(beta[0]*(modulated_voigt(x-q[0],*q[1:],nodes=256)
                                               -modulated_voigt(x-q[0],*q[1:],nodes=128)))))
    if validation_error>1e-6:
        raise RuntimeError(f"Final quadrature resolution insufficient: {validation_error}")
    summary={
        "mean_model":"finite-modulation Voigt + linear baseline; unweighted LS; 7 mean parameters",
        "reference_Hz":ref,"shape_parameters_kHz":dict(zip(["center","gamma","sigma","modulation_peak"],q.tolist())),
        "linear_parameters":dict(zip(["amplitude","offset","slope_per_kHz"],beta.tolist())),
        "SSE":sse,"RMS":float(np.sqrt(sse/len(y))),"core_RMS":float(np.sqrt(np.mean(residual[core]**2))),
        "core_lag1":acf(residual[core]),"n":len(y),"core_n":int(sum(core)),
        "quadrature_128_vs_256_max_signal_difference":validation_error,
        "selected_conditional_noise_model":best_name,"noise_parameters":p,
        "noise_frequency_scaling":{"mean_offset_kHz":xmean,"std_kHz":xstd},
        "mean_parameter_bounds_kHz":{"center":[-5,5],"gamma":[.02,5],"sigma":[0,3],"modulation_peak":[0,3]},
        "baseline_left_SD":float(residual[x < -10].std(ddof=1)),
        "baseline_right_SD":float(residual[x > 10].std(ddof=1)),
        "baseline_lag1_by_side":[acf(residual[x < -10]),acf(residual[x > 10])],
        "limitations":[
            "Noise models condition on the estimated mean; neither mean uncertainty nor joint MLE is included.",
            "Conditional score=2*NLL+2*noise_parameter_count, only an exploratory comparison; no significance claim.",
            "Profile reference deviance assumes iid constant-variance noise; no confidence interval is asserted.",
            "Near zero modulation the profile is flat; a tiny positive optimizer value is not a measured modulation amplitude.",
            "AR(1) rho is per row, not seconds or frequency distance; sampling times are unknown.",
            "Correlated residuals may be deterministic mean-model error, not random noise.",
            "Gaussian static spread and small per-point jitter are effective descriptions, not identified sources.",
            "No sweep lag, full Bloch dynamics, magnetic field calibration, or intrinsic T2 extraction.",
        ],
    }
    Path(str(prefix)+"_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    fig,(ax,aw)=plt.subplots(2,1,figsize=(9,7),sharex=True)
    ax.plot(profile.modulation_peak_kHz,profile.iid_reference_delta_deviance,"o-",color="#245a9b")
    ax.set_ylabel("N log(SSE / minimum SSE)")
    ax.set_title("Broadening trade-off: refit all other mean parameters at each modulation amplitude")
    aw.plot(profile.modulation_peak_kHz,profile.sigma_kHz,"o-",label="Gaussian SD")
    aw.plot(profile.modulation_peak_kHz,profile.gamma_kHz,"s-",label="Lorentz HWHM")
    aw.set(xlabel="Modulation peak excursion (kHz)",ylabel="Fitted width (kHz)")
    aw.legend()
    for axis in (ax,aw):axis.grid(alpha=.25)
    fig.text(.5,.01,"Top panel is an iid-noise reference scale, not a confidence interval.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(str(prefix)+"_profile.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    mask=abs(x)<7
    axes[0].scatter(x[mask],y[mask],s=15,c="black",alpha=.6,label="Measured")
    xx=np.linspace(-7,7,1500)
    yy=beta[0]*modulated_voigt(xx-q[0],*q[1:],nodes=128)+beta[1]+beta[2]*xx
    axes[0].plot(xx,yy,color="#245a9b",label="Finite-modulation Voigt")
    axes[0].legend();axes[0].set_ylabel("Signal (original units)")
    axes[0].set_title("Mean fit and conditional residual-noise diagnostics")
    axes[1].plot(x[mask],residual[mask],".-",lw=.7,label="Residual")
    marginal=np.sqrt(white**2+(p["jitter"]*derivative)**2+(p["gain"]*signal)**2)
    axes[1].fill_between(x[mask],-marginal[mask],marginal[mask],color="#245a9b",alpha=.2,
                         label="Fitted marginal +/-1 SD (conditional)")
    axes[1].set_ylabel("Residual");axes[1].legend(fontsize=8)
    axes[2].plot(x[mask],z[mask],".-",color="#a04a21",lw=.7)
    axes[2].axhline(2,color="gray",ls=":");axes[2].axhline(-2,color="gray",ls=":")
    axes[2].set(ylabel="Standardized innovation",xlabel=f"Frequency offset from {ref/1e6:.6f} MHz (kHz)")
    for axis in axes:axis.axhline(0,color="black",lw=.5);axis.grid(alpha=.2)
    fig.text(.5,.01,"Noise fit is conditional on the mean; correlated model discrepancy can imitate random fluctuations.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(str(prefix)+"_diagnostics.png",dpi=170);plt.close(fig)
    print(f"\n平均波形の共鳴周波数: {(ref+1000*q[0])/1e6:.8f} MHz")
    print(f"残差RMS: 全体={summary['RMS']:.6f}, 共鳴付近={summary['core_RMS']:.6f}")
    print("候補中で条件付き比較スコアが最小のノイズモデル:", best_name)
    print("詳しいパラメータ・仮定・図の保存先:", prefix, flush=True)
    if args.bootstrap:
        # 6. 必要なら、このモデルから模擬測定を繰り返して不確かさを求める。
        from nmr_uncertainty import run_uncertainty
        run_uncertainty(src,replicates=args.bootstrap,workers=args.workers,seed=args.seed)


if __name__=="__main__":
    main()
