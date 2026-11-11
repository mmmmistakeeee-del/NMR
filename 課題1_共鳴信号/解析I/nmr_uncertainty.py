"""NMR解析の不確かさを、模擬測定と再フィットで求める。

保存済みの300回分を読み、結果を確認する（模擬測定は実行しない）:
    python -B nmr_uncertainty.py --summarize-only

模擬測定から計算し直す:
    python -B nmr_uncertainty.py --replicates 300 --workers 4

読む順序:
    load_saved_analysis → run_bootstrap_refits → save_bootstrap_results

計算の意味:
    1. 前段で推定した「平均波形＋相関ノイズ」を読み込む。
    2. 同じ周波数点で測ったつもりの模擬データを何回も作る。
    3. 毎回フィットし、推定値がどのくらい散らばるかを調べる。

bootstrap_SD は再フィット値の標準偏差。試行回数の平方根では割らない。
仮定したモデル内のばらつきであり、校正誤差などは含まない。
幅の分離が難しいため、百分位範囲を厳密な95%信頼区間とは扱わない。

変数の共通表記（下位の数値関数との互換性のため残している）:
    x: 基準周波数からの差 [kHz] / y: 測定信号 [元データ単位]
    q: [中心の差, ローレンツ半値半幅, ガウス分布幅, 変調片振幅] [kHz]
    beta: [信号の倍率, 背景の切片, 背景の傾き]
    ref: 基準周波数 [Hz] / pred: フィット曲線の各測定点での値
"""

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import voigt_profile

from nmr_joint_noise import fit_mean, fit_noise, modulated_voigt


_CONTEXT = None

# CSVの英語識別子は従来どおり残し、日本語表示を別に用意する。
PARAMETER_LABELS = {
    "f0_MHz": "共鳴周波数",
    "gamma_HWHM_kHz": "ローレンツ半値半幅",
    "static_sigma_kHz": "共鳴周波数分布の幅",
    "modulation_peak_kHz": "変調の片側周波数振幅",
    "observed_peak_to_peak_kHz": "観測信号の山と谷の間隔",
    "absorption_Voigt_FWHM_kHz": "変調前の吸収曲線の半値全幅",
    "amplitude": "信号の倍率",
    "baseline_offset": "背景の切片",
    "baseline_slope_per_kHz": "背景の傾き",
    "white_SD_at_ref": "共鳴付近の電圧雑音の標準偏差",
    "noise_log_SD_trend": "電圧雑音の大きさの周波数依存係数",
    "jitter_SD_Hz": "周波数揺らぎの標準偏差",
    "gain_SD_percent": "相対振幅揺らぎの標準偏差",
    "latent_rho_per_observation": "相関成分の隣接点間相関",
}


def simulate_noise(rng, white, hf, hg, rho, count=1):
    """独立な電圧雑音と、隣接点で相関する2成分を生成する。

    white: 各点の電圧雑音の標準偏差
    hf: 周波数揺らぎの大きさ × 平均信号の傾き
    hg: 相対振幅揺らぎの大きさ × 平均信号
    rho: 1測定点離れた相関（時間間隔は不明なので秒には換算しない）
    u/g は分散1。rhoだけ前の値を引き継ぎ、残りを新しい乱数で補う。
    """
    n = len(white)
    e = rng.normal(size=(count, n))*white
    u = rng.normal(size=count)
    g = rng.normal(size=count)
    innovation_sd = np.sqrt(1-rho*rho)
    e[:, 0] += hf[0]*u+hg[0]*g
    for i in range(1, n):
        u = rho*u+innovation_sd*rng.normal(size=count)
        g = rho*g+innovation_sd*rng.normal(size=count)
        e[:, i] += hf[i]*u+hg[i]*g
    return e


def validate_simulator():
    rng = np.random.default_rng(921503)
    white = np.linspace(.1, .2, 8)
    hf = np.linspace(-.15, .15, 8)
    hg = .12*np.cos(np.arange(8))
    rho = .73
    samples = simulate_noise(rng, white, hf, hg, rho, count=50000)
    distance = abs(np.arange(8)[:, None]-np.arange(8)[None, :])
    expected = np.diag(white**2)+(np.outer(hf,hf)+np.outer(hg,hg))*rho**distance
    observed = np.cov(samples, rowvar=False)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    error = float(np.max(abs(observed-expected)/scale))
    assert error < .035, error
    print(f"雑音生成器の共分散を確認しました（規格化した最大誤差 {error:.4f}）。", flush=True)


def refit(x, y, q0):
    """模擬データ1系列を再フィットし、数値積分の精度も確認する。

    通常の解、変調幅ゼロの極限、ガウス幅ゼロの極限を比較する。
    戻り値: ((二乗残差和, q, beta, 各点の曲線値), 積分精度の確認値)
    """
    starts = [q0, [q0[0], .55, .8, 1.], [q0[0], .8, .4, 1.4]]
    candidates = [fit_mean(x,y,starts,nodes=48),
                  fit_mean(x,y,[q0],fixed_a=0.,nodes=48),
                  fit_mean(x,y,[[q0[0],1.,0.,1.3]],fixed_sigma=0.,nodes=48)]
    best = min(candidates,key=lambda v:v[0])
    # Select the exact small-modulation boundary when the SSEs are numerically tied.
    if candidates[1][0] <= best[0]+1e-8:
        best = candidates[1]
    sse,q,beta,pred=best
    check=beta[0]*modulated_voigt(x-q[0],*q[1:],nodes=128)+beta[1]+beta[2]*x
    error=float(np.max(abs(check-pred)))
    if error>2e-6:
        best=fit_mean(x,y,[q,*starts],nodes=128)
        boundary=fit_mean(x,y,[q],fixed_a=0.,nodes=128)
        best=min([best,boundary,candidates[2]],key=lambda v:v[0])
        sse,q,beta,pred=best
        check=beta[0]*modulated_voigt(x-q[0],*q[1:],nodes=256)+beta[1]+beta[2]*x
        error=float(np.max(abs(check-pred)))
        if error>2e-6:
            raise RuntimeError(f"Quadrature not resolved: {error}")
    return best,error


def quantities(q,beta,ref):
    """内部のq/betaを、中心周波数・線幅などの報告する量に換算する。

    山と谷の間隔: 背景を除いた「変調後の観測信号」で計算。
    吸収の半値全幅: 「変調前のVoigt吸収曲線」で計算。
    """
    gamma,sigma,a=q[1:]
    maximum=voigt_profile(0.,sigma,gamma)
    halfwidth=brentq(lambda u: voigt_profile(u,sigma,gamma)-maximum/2,
                     0.,20*(gamma+sigma))
    peak=minimize_scalar(lambda u: float(modulated_voigt(np.array([u]),gamma,sigma,a,nodes=128)[0]),
                         bounds=(0.,10*(gamma+sigma+a)),method="bounded",
                         options={"xatol":1e-8})
    if not peak.success:
        raise RuntimeError("Observed peak-spacing calculation failed")
    return dict(f0_MHz=(ref+1000*q[0])/1e6,gamma_HWHM_kHz=gamma,
                static_sigma_kHz=sigma,modulation_peak_kHz=a,
                observed_peak_to_peak_kHz=2*peak.x,absorption_Voigt_FWHM_kHz=2*halfwidth,
                amplitude=beta[0],baseline_offset=beta[1],baseline_slope_per_kHz=beta[2])


def noise_quantities(p,xmean,xstd):
    """ノイズの内部パラメータを、信号単位・Hz・%に換算する。"""
    # With no colored component its correlation has no identifiable value.
    rho = p["rho"] if p["gain"]>=1e-7 or p["jitter"]>=1e-6 else np.nan
    return dict(white_SD_at_ref=np.exp(p["log_sigma"]-p["trend"]*xmean/xstd),
                noise_log_SD_trend=p["trend"],jitter_SD_Hz=1000*p["jitter"],
                gain_SD_percent=100*p["gain"],latent_rho_per_observation=rho)


def initialize_worker(context):
    """並列ワーカーへ共通の入力を1回だけ渡す。物理モデルの計算ではない。"""
    global _CONTEXT
    _CONTEXT=context


def one_replicate(index):
    """模擬測定を1回行い、平均波形と選択済みノイズモデルを再推定する。

    contextはload_saved_analysisで作る。乱数は試行番号ごとに固定するので、
    並列ワーカーの数を変えても同じ番号の模擬測定は同じになる。
    """
    c=_CONTEXT
    rng=np.random.default_rng(np.random.SeedSequence([c["seed"],index]))
    sample=c["mean"]+simulate_noise(rng,c["white"],c["hf"],c["hg"],c["rho"])[0]
    try:
        (sse,q,beta,pred),error=refit(c["x"],sample,c["q"])
        row=dict(replicate=index,status="ok",SSE=sse,quadrature_error=error)
        row.update(quantities(q,beta,c["ref"]))
        row.update(gamma_lower_boundary=q[1]<=.02001, sigma_lower_boundary=q[2]<=1e-5,
                   modulation_lower_boundary=q[3]<=1e-5,any_upper_boundary=bool(
                       abs(q[0])>=4.999 or q[1]>=4.999 or q[2]>=2.999 or q[3]>=2.999))
        if c["refit_noise"]:
            try:
                signal=pred-beta[1]-beta[2]*c["x"]
                h=1e-4
                derivative=beta[0]*(modulated_voigt(c["x"]+h-q[0],*q[1:],nodes=128)
                                   -modulated_voigt(c["x"]-h-q[0],*q[1:],nodes=128))/(2*h)
                p,_,_,_,_=fit_noise(c["noise_name"],sample-pred,c["xscaled"],signal,derivative)
                row.update(noise_quantities(p,c["xmean"],c["xstd"]))
                row["noise_status"]="ok"
            except Exception as exc:
                row["noise_status"]=str(exc)
        return row
    except Exception as exc:
        return dict(replicate=index,status="error",error=str(exc))


def load_saved_analysis(src, seed=20261028, refit_noise=True):
    """1. 元データと前段の解析結果を照合し、模擬測定の入力を準備する。

    contextには測定点x、元データy、平均波形mean、q/beta、ノイズの
    大きさwhite/hf/hg、隣接相関rho、乱数シードなどをまとめて入れる。
    """
    src=Path(src).resolve()
    prefix=src.with_name(src.stem+"_joint_noise")
    summary=json.loads(Path(str(prefix)+"_summary.json").read_text(encoding="utf-8"))
    saved=pd.read_csv(str(prefix)+"_residuals.csv")
    f,y=np.loadtxt(src).T
    if len(saved)!=len(f) or not np.allclose(saved.frequency_Hz,f,rtol=0,atol=1e-8) or not np.allclose(saved.signal,y,rtol=0,atol=1e-12):
        raise ValueError("Saved analysis does not match current data; rerun nmr_joint_noise.py")
    ref=summary["reference_Hz"]; x=(f-ref)/1000
    q=np.array([summary["shape_parameters_kHz"][k] for k in ["center","gamma","sigma","modulation_peak"]])
    beta=np.array([summary["linear_parameters"][k] for k in ["amplitude","offset","slope_per_kHz"]])
    signal=beta[0]*modulated_voigt(x-q[0],*q[1:],nodes=128)
    mean=signal+beta[1]+beta[2]*x
    if not np.allclose(mean,saved.mean_fit,rtol=0,atol=1e-6):
        raise ValueError("Saved parameters and residual file disagree")
    p=summary["noise_parameters"]
    xmean=summary["noise_frequency_scaling"]["mean_offset_kHz"]
    xstd=summary["noise_frequency_scaling"]["std_kHz"]
    xscaled=(x-xmean)/xstd
    white=np.exp(p["log_sigma"]+p["trend"]*xscaled)
    h=1e-4
    derivative=beta[0]*(modulated_voigt(x+h-q[0],*q[1:],nodes=128)
                       -modulated_voigt(x-h-q[0],*q[1:],nodes=128))/(2*h)
    context=dict(x=x,mean=mean,q=q,ref=ref,white=white,hf=p["jitter"]*derivative,hg=p["gain"]*signal,
                 rho=p["rho"],seed=seed,xmean=xmean,xstd=xstd,xscaled=xscaled,
                 refit_noise=refit_noise,noise_name=summary["selected_conditional_noise_model"])
    context.update(y=y, beta=beta, p=p)
    return src, context


def run_bootstrap_refits(context, replicates, workers, output):
    """2. 模擬測定と再フィットを繰り返し、各試行の結果を保存する。

    数値処理はone_replicateを読む。ここは並列実行と進捗保存の管理。
    """
    rows=[]; start=time.monotonic()
    if workers==1:
        initialize_worker(context)
        iterator=(one_replicate(i) for i in range(replicates))
        executor=None
    else:
        executor=ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(context,))
        futures=[executor.submit(one_replicate,i) for i in range(replicates)]
        iterator=(future.result() for future in as_completed(futures))
    try:
        for row in iterator:
            rows.append(row)
            if len(rows)%10==0 or len(rows)==replicates:
                pd.DataFrame(rows).sort_values("replicate").to_csv(str(output)+"_samples.csv",index=False)
                failures=sum(v["status"]!="ok" for v in rows)
                print(f"模擬測定と再フィット: {len(rows)}/{replicates}回; 失敗={failures}回; "
                      f"経過={time.monotonic()-start:.0f}秒",flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    samples=pd.DataFrame(rows).sort_values("replicate")
    return samples


def summarize_parameters(good, context):
    """3. 各パラメータの標準偏差・中央値・百分位点・偏りを集計する。

    標準偏差は values.std(ddof=1)。模擬測定を何回に増やしても、
    実験の不確かさを1/sqrt(試行回数)に縮める計算はしない。
    """
    q, beta, ref = context["q"], context["beta"], context["ref"]
    p, xmean, xstd = context["p"], context["xmean"], context["xstd"]
    refit_noise = context["refit_noise"]
    estimates=quantities(q,beta,ref)
    if refit_noise:
        estimates.update(noise_quantities(p,xmean,xstd))
    units=dict(f0_MHz="MHz",gamma_HWHM_kHz="kHz",static_sigma_kHz="kHz",modulation_peak_kHz="kHz",
               observed_peak_to_peak_kHz="kHz",absorption_Voigt_FWHM_kHz="kHz",amplitude="signal*kHz^2",
               baseline_offset="signal",baseline_slope_per_kHz="signal/kHz",white_SD_at_ref="signal",
               noise_log_SD_trend="dimensionless",jitter_SD_Hz="Hz",gain_SD_percent="percent",
               latent_rho_per_observation="dimensionless")
    stats=[]
    for name,estimate in estimates.items():
        values=good[name].dropna() if name in good else pd.Series(dtype=float)
        if len(values)<2:
            continue
        note="conditional parametric bootstrap"
        if name in ("modulation_peak_kHz","static_sigma_kHz","gamma_HWHM_kHz"):
            note="weakly identified/correlated widths; percentile range is descriptive, not calibrated confidence interval"
        if name=="jitter_SD_Hz" and "jitter" not in context["noise_name"] and "both" not in context["noise_name"]:
            note="fixed to zero by selected noise model; zero SD does not establish absence of jitter"
        if name=="latent_rho_per_observation":
            note="conditional on nonzero fitted colored-noise amplitude; undefined zero-amplitude cases excluded"
        quantiles=np.quantile(values,[.025,.16,.5,.84,.975])
        stats.append(dict(parameter=name,unit=units[name],estimate=estimate,
                          bootstrap_SD=values.std(ddof=1),bootstrap_mean=values.mean(),
                          bootstrap_bias=values.mean()-estimate,q025=quantiles[0],q16=quantiles[1],
                          median=quantiles[2],q84=quantiles[3],q975=quantiles[4],n_success=len(values),note=note))
    table=pd.DataFrame(stats)
    return table, estimates


def build_metadata(samples, good, context, src, replicates, workers, seed, refit_noise):
    """乱数シード・成功数・境界の割合など、再現に必要な条件を記録する。"""
    rows = samples.to_dict("records")
    metadata=dict(method="Parametric bootstrap of the full two-stage estimator; same noise-model structure",
                  replicates_requested=replicates,mean_successes=len(good),seed=seed,workers=workers,
                  noise_successes=int((good.get("noise_status",pd.Series(dtype=str))=="ok").sum()),
                  rho_defined_refits=int(good.get("latent_rho_per_observation",pd.Series(dtype=float)).notna().sum()),
                  noise_model=context["noise_name"],noise_parameters_refitted=refit_noise,
                  source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  boundary_fractions={k:float(good[k].mean()) for k in ["gamma_lower_boundary","sigma_lower_boundary",
                                       "modulation_lower_boundary","any_upper_boundary"]},
                  failures=[r for r in rows if r["status"]!="ok"],
                  sd_definition="sample standard deviation across fitted bootstrap estimates, ddof=1; no division by sqrt(B)",
                  limitations=["Conditional on fitted generating noise parameters and mean family; no model-selection uncertainty",
                               "Nonregular/boundary widths: percentile ranges are descriptive; nominal coverage not established",
                               "No frequency calibration, sweep-lag, modulation-calibration or missing-physics systematics",
                               "Noise SD is different from uncertainty of the estimated noise SD",
                               "AR correlation is per observation, not seconds",
                               "Percentile endpoints have finite Monte Carlo error; increase replicates for more stable endpoints"])
    analysis_file = src.with_name(src.stem + "_joint_noise_summary.json")
    metadata["analysis_sha256"] = hashlib.sha256(analysis_file.read_bytes()).hexdigest()
    return metadata


def plot_parameter_distributions(good, estimates, q, output):
    """4a. 中心周波数・線幅の分布と、幅の交換関係を図にする。"""
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    center_est=estimates["f0_MHz"]
    axes[0,0].hist((good.f0_MHz-center_est)*1e6,bins=25,color="#245a9b",alpha=.8)
    axes[0,0].axvline(0,color="black",lw=1)
    axes[0,0].set(xlabel="Fitted resonance - original estimate (Hz)",ylabel="Bootstrap count")
    axes[0,1].hist(good.observed_peak_to_peak_kHz,bins=25,color="#245a9b",alpha=.8)
    axes[0,1].axvline(estimates["observed_peak_to_peak_kHz"],color="black",lw=1)
    axes[0,1].set(xlabel="Observed signal peak-to-peak width (kHz)",ylabel="Bootstrap count")
    axes[1,0].scatter(good.modulation_peak_kHz,good.static_sigma_kHz,s=12,alpha=.45)
    axes[1,0].scatter([q[3]],[q[2]],marker="*",s=130,c="#b34a24",label="Original fit")
    axes[1,0].set(xlabel="Modulation peak (kHz)",ylabel="Static Gaussian SD (kHz)")
    axes[1,0].legend()
    axes[1,1].hist(good.gamma_HWHM_kHz,bins=25,color="#245a9b",alpha=.8)
    axes[1,1].axvline(q[1],color="black",lw=1)
    axes[1,1].set(xlabel="Lorentz HWHM (kHz)",ylabel="Bootstrap count")
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle(f"Parameter uncertainty: {len(good)} correlated-noise bootstrap refits")
    fig.text(.5,.01,"Conditional simulation distributions; boundary widths do not imply calibrated confidence intervals.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96)); fig.savefig(str(output)+"_distributions.png",dpi=160);plt.close(fig)



def plot_mean_uncertainty(good, context, output):
    """4b. 再フィットされた平均曲線の16–84%範囲を図にする。

    個々の測定値の散らばりとは違う。平均波形を推定する不確かさを示す。
    """
    x, y, ref = context["x"], context["y"], context["ref"]
    q, beta = context["q"], context["beta"]
    xx=np.linspace(-6,6,400)
    curves=[]
    for _,r in good.iterrows():
        center=(r.f0_MHz*1e6-ref)/1000
        curves.append(r.amplitude*modulated_voigt(xx-center,r.gamma_HWHM_kHz,r.static_sigma_kHz,
                                                r.modulation_peak_kHz,nodes=128)
                      +r.baseline_offset+r.baseline_slope_per_kHz*xx)
    curves=np.array(curves)
    low,high=np.quantile(curves,[.16,.84],axis=0)
    central=beta[0]*modulated_voigt(xx-q[0],*q[1:],nodes=128)+beta[1]+beta[2]*xx
    pd.DataFrame(dict(offset_kHz=xx,original_mean=central,bootstrap_mean=curves.mean(axis=0),
                      bootstrap_SD=curves.std(axis=0,ddof=1),q16=low,q84=high)).to_csv(str(output)+"_mean_band.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5))
    mask=abs(x)<6
    ax.scatter(x[mask],y[mask],s=15,c="black",alpha=.5,label="Measured")
    ax.fill_between(xx,low,high,color="#245a9b",alpha=.25,label="16-84% refitted-mean percentile band")
    ax.plot(xx,central,color="#245a9b",label="Original mean fit")
    ax.set(xlabel=f"Frequency offset from {ref/1e6:.6f} MHz (kHz)",ylabel="Signal (original units)",
           title="Uncertainty of the fitted mean (not a prediction band for individual measurements)")
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(str(output)+"_mean_band.png",dpi=160);plt.close(fig)


def print_parameter_summary(table, metadata):
    """最初に読むとよい量を、日本語名と単位で画面に表示する。"""
    print(f"\n再フィット成功: {metadata['mean_successes']}/{metadata['replicates_requested']} 回")
    print("標準偏差は、仮定した相関ノイズモデルの下での標準不確かさです。")
    for name in ("f0_MHz", "observed_peak_to_peak_kHz", "gamma_HWHM_kHz", "static_sigma_kHz"):
        row = table.loc[table.parameter == name].iloc[0]
        extra = f"（{row.bootstrap_SD * 1e6:.2f} Hz）" if name == "f0_MHz" else ""
        print(f"  {PARAMETER_LABELS[name]}: {row.estimate:.9g} {row.unit}; "
              f"標準偏差 {row.bootstrap_SD:.4g} {row.unit}{extra}")
    fraction = metadata["boundary_fractions"]["modulation_lower_boundary"]
    print(f"変調幅がゼロ境界となった割合: {fraction:.1%}。幅は分布図も確認してください。")


def save_bootstrap_results(src, context, samples, metadata):
    """集計表、日本語の集計表、相関行列、図を保存する共通の出口。"""
    output = src.with_name(src.stem + "_uncertainty")
    samples = samples.copy().sort_values("replicate")
    if samples.replicate.duplicated().any():
        raise ValueError("同じ試行番号が重複しています。保存データを確認してください。")
    # 相関成分がゼロならrhoは決められない。保存済みの古い結果にも適用する。
    if {"gain_SD_percent", "jitter_SD_Hz"}.issubset(samples.columns):
        undefined = (samples.gain_SD_percent < 1e-5) & (samples.jitter_SD_Hz < 1e-3)
        samples.loc[undefined, "latent_rho_per_observation"] = np.nan
    samples.to_csv(str(output) + "_samples.csv", index=False)
    good = samples[samples.status == "ok"]
    requested = metadata["replicates_requested"]
    if len(good) < max(2, .9 * requested):
        raise RuntimeError("平均波形の再フィット成功率が90%未満のため、集計を停止しました。")

    table, estimates = summarize_parameters(good, context)
    table.to_csv(str(output) + "_parameters.csv", index=False)
    columns = {"parameter": "識別子", "unit": "単位", "estimate": "元データの推定値",
               "bootstrap_SD": "標準不確かさ_標準偏差", "bootstrap_mean": "再推定値の平均",
               "bootstrap_bias": "再推定平均と元推定値の差", "q025": "2.5パーセンタイル",
               "q16": "16パーセンタイル", "median": "中央値", "q84": "84パーセンタイル",
               "q975": "97.5パーセンタイル", "n_success": "集計できた回数", "note": "注意事項"}
    japanese = table.rename(columns=columns)
    japanese.insert(0, "量", table.parameter.map(PARAMETER_LABELS))
    japanese["注意事項"] = table.note.replace({
        "conditional parametric bootstrap": "仮定した平均・ノイズモデルに基づく再推定のばらつき",
        "weakly identified/correlated widths; percentile range is descriptive, not calibrated confidence interval":
            "幅どうしが補い合うため分離しにくい。百分位範囲は再推定分布の範囲",
        "fixed to zero by selected noise model; zero SD does not establish absence of jitter":
            "選択モデルでゼロに固定。実際に周波数揺らぎがないと証明した意味ではない",
        "conditional on nonzero fitted colored-noise amplitude; undefined zero-amplitude cases excluded":
            "相関成分の振幅が非ゼロの試行だけ集計。振幅ゼロでは相関係数を決められない",
    })
    japanese["単位"] = table.unit.replace({
        "signal": "元データの信号単位", "signal*kHz^2": "信号単位×kHz²",
        "signal/kHz": "信号単位/kHz", "dimensionless": "無次元", "percent": "%",
    })
    japanese.to_csv(str(output) + "_parameters_ja.csv", index=False, encoding="utf-8-sig")
    available = [name for name in estimates if name in good]
    good[available].corr().to_csv(str(output) + "_correlation.csv")
    metadata["mean_successes"] = len(good)
    metadata["rho_defined_refits"] = int(good.get("latent_rho_per_observation", pd.Series(dtype=float)).notna().sum())
    Path(str(output) + "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_parameter_distributions(good, estimates, context["q"], output)
    plot_mean_uncertainty(good, context, output)
    print_parameter_summary(table, metadata)
    print("日本語の結果表:", str(output) + "_parameters_ja.csv", flush=True)
    return table, metadata


def run_uncertainty(src, replicates=300, workers=4, seed=20261028, refit_noise=True):
    """読み込み → 模擬測定と再フィット → 集計・作図の順に実行する。"""
    if replicates < 2 or workers < 1:
        raise ValueError("試行回数は2以上、ワーカー数は1以上にしてください。")
    src, context = load_saved_analysis(src, seed, refit_noise)
    validate_simulator()
    output = src.with_name(src.stem + "_uncertainty")
    samples = run_bootstrap_refits(context, replicates, workers, output)
    good = samples[samples.status == "ok"]
    metadata = build_metadata(samples, good, context, src, replicates, workers, seed, refit_noise)
    return save_bootstrap_results(src, context, samples, metadata)


def summarize_existing(src):
    """保存済みの試行を集計し直す。模擬測定・再フィットは行わない。"""
    src = Path(src).resolve()
    output = src.with_name(src.stem + "_uncertainty")
    metadata = json.loads(Path(str(output) + "_metadata.json").read_text(encoding="utf-8"))
    if metadata["source_sha256"] != hashlib.sha256(src.read_bytes()).hexdigest():
        raise ValueError("元データが変わっています。不確かさを計算し直してください。")
    src, context = load_saved_analysis(src, metadata["seed"], metadata["noise_parameters_refitted"])
    analysis_file = src.with_name(src.stem + "_joint_noise_summary.json")
    analysis_hash = hashlib.sha256(analysis_file.read_bytes()).hexdigest()
    if metadata.get("analysis_sha256", analysis_hash) != analysis_hash:
        raise ValueError("平均波形・ノイズの解析結果が変わっています。不確かさを計算し直してください。")
    if metadata["noise_model"] != context["noise_name"]:
        raise ValueError("保存された模擬測定と、現在のノイズモデルが一致しません。")
    metadata["analysis_sha256"] = analysis_hash
    samples = pd.read_csv(str(output) + "_samples.csv")
    if len(samples) != metadata["replicates_requested"]:
        raise ValueError("保存された試行が揃っていません。完了した計算の結果を指定してください。")
    print("保存済みの試行を集計します。模擬測定の再計算はしません。", flush=True)
    return save_bootstrap_results(src, context, samples, metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default=str(Path(__file__).with_name("20261020_KITtest01.dat")),
                        help="周波数[Hz]・信号の2列ファイル。省略時は同じフォルダの測定データ")
    parser.add_argument("--replicates", type=int, default=300, help="模擬測定の回数（既定: 300）")
    parser.add_argument("--workers", type=int, default=4, help="並列で計算する数（既定: 4）")
    parser.add_argument("--seed", type=int, default=20261028, help="乱数の再現用番号")
    parser.add_argument("--skip-noise-refit", action="store_true", help="各試行でノイズ強度の再推定を省略する")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="雑音生成器の数値検証だけを行う")
    mode.add_argument("--summarize-only", action="store_true", help="保存済みの試行から集計表と図を作る（再フィットなし）")
    args=parser.parse_args()
    if args.validate_only:
        validate_simulator()
        return
    if args.summarize_only:
        summarize_existing(args.file)
        return
    run_uncertainty(args.file,args.replicates,args.workers,args.seed,not args.skip_noise_refit)


if __name__=="__main__":
    main()
