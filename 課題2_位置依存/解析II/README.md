# NMR 実験II：微分ローレンツ一括解析

## 実行

Python 3.10以上と NumPy / SciPy / Matplotlib を使用します。
このフォルダで実行する場合：

```powershell
python -m pip install -r requirements.txt
python fit_nmr_batch.py
python plot_nmr_results.py
```

既定の入力は、このフォルダの隣にある `../実験II`、出力は `results` です。
作業ディレクトリによらずスクリプトの位置を基準に解決します。
データを追加したら上記2つのスクリプトを再実行します。CSVは全件再計算して上書きします。
確認図は同名を上書きしますが、削除した入力の古い図は消しません。今回の出力一覧はCSVの
`diagnostic_plot`列が基準です。別条件を比較するときは別の出力先を指定してください。

```powershell
python fit_nmr_batch.py "../実験II" --output results_constant --baseline constant
python plot_nmr_results.py results_constant/fit_results.csv
```

出力先を明示して、任意の場所にあるデータにも使えます。

```powershell
python fit_nmr_batch.py "測定データのフォルダ" --output results_new
python plot_nmr_results.py results_new/fit_results.csv --output results_new/plots
```

## 11月4日と今後の位置測定

[11月4日の位置測定](../測定記録/20261104/)は z=4.0 の4座標・5ファイル。x=6.0は、周波数刻みを細かくして取り直した `_2` を採用する。[再測定の採用ルールと引き継ぎ](../測定と解析の引き継ぎ.md)・[11月4日の位置依存図](../../課題3_変調条件と追加測定/解析III/results/plots/position_dependence.png)

今後の z < 4.5 のデータも、`測定記録` の下に `YYYYMMDD` 形式の測定日フォルダを作って保存する。同じ日の追測はその日のフォルダに加え、再測定の識別子をファイル名に残す。解析時は測定日フォルダと専用の出力先を指定する。以下は11月4日分の例。

```powershell
python fit_nmr_batch.py "../測定記録/20261104" --output results_20261104
python plot_nmr_results.py results_20261104/fit_results.csv
```

既存分・11月4日分・11月10日分をまとめた xz 図を作るには、次を実行する。追加の測定日も含める場合は `--additional-input` を測定日フォルダごとに指定する。[11月10日の測定](../測定記録/20261110/)の直下には通常測定37件を置き、Delay Timeの試行はサブフォルダに分けている。

```powershell
python fit_nmr_batch.py --additional-input "../測定記録/20261104" --additional-input "../測定記録/20261110" --output results_with_additions --no-plots
python plot_nmr_results.py results_with_additions/fit_results.csv
```

[追加分を含む共鳴周波数のxz図](results_with_additions/plots/f0_Hz_xz.png)には各日の採用値を表示する。測定日間の周波数差を解釈するときは、測定条件と座標の対応を確認する。

## 入力

- ヘッダーなし・空白/タブ区切りの2列：周波数、測定信号。#コメントは可。
- 周波数は既定でHz。kHz/MHzなら `--unit kHz` / `--unit MHz` を指定。
- ファイル名は `YYYYMMDD_KIT_x2.5_z4.5_3.dat` 等。末尾の識別子は自由。
- 直下の `.dat` を全件処理。サブフォルダは検索しません。
- 同じ座標ではファイル名の測定日が最新のものを採用する。同じ日は末尾の再測定番号が最大のものを採用する。比較で外れたファイルは `run_summary.json` の `superseded_files` に記録する。
- 元データは読み取るだけで、変更しません。
- NaN/Inf行は除去し、数を記録。他の形式不正やファイル名不正はfailed行になります。
- 取得順序で掃引を分割し、昇り/下りを別行で解析。折返し点は前の区間にのみ含めます。
  連続する同一周波数は保持します。小さな逆行も区切りになるので警告を確認してください。
- 1区間12点以上・異なる周波数8点以上が必要です。

## 関数と線幅の定義

```text
y(f) = -2 Gamma^2 L0 (f-f0) / [Gamma^2+(f-f0)^2]^2
       + b0 + b1 (f-f_ref)
```

- `f0_Hz`：共鳴周波数。
- `hwhm_Hz = Gamma`：元の吸収ローレンツ線の半値半幅。
- `fwhm_Hz = 2 Gamma`：元の吸収線の半値全幅。
- `peak_to_peak_Hz = 2 Gamma / sqrt(3)`：微分信号の山と谷の周波数間隔。
- `L0`：振幅係数。信号が吸収線の微分に比例する場合、装置の比例係数を含む。
  その単位は「測定信号の単位 × Hz」。正負を許します。
- `baseline`：f_refでの背景値。単位は測定信号と同じ。
- `baseline_slope_per_Hz`：背景の傾き。単位は測定信号/Hz。
- f_refは各解析区間の周波数範囲の中点です。

既定は直線背景。`--baseline constant`では傾きを0に固定します。
`--fmin 16000000 --fmax 16030000` のようにフィット範囲をHzで指定できます。
範囲制限は掃引区間の分割後に適用します。

## 計算方法と誤差

周波数を中心化・無次元化し、複数の初期値から二乗残差和を最小化します。
周波数中心とlog(Gamma)を非線形最適化し、各試行で振幅と背景を線形最小二乗で求めます
（variable projection）。平滑化は初期値の推定のみに使用し、フィットには生データを使います。

中心は測定範囲内、Gammaは代表周波数間隔の0.1倍以上かつ測定幅の2倍以下に制約します。
境界に近い結果は警告扱いです。これは物理的に正しい共鳴が必ず見つかる保証ではありません。

全自由パラメータのヤコビアンから、残差分散 SSE/(N-p) でスケールした共分散を求めます。
`*_se*`は標準誤差（1 sigma）。周波数校正・磁場ドリフト・モデルのずれ等の系統誤差は含みません。
各点の誤差が不明なため、独立・等分散の誤差を仮定します。
残差に相関や構造がある場合、この標準誤差は過小評価になり得ます。

内部では A=L0/Gamma を使いますが、L0の誤差にはGammaとの相関も含めて変換します。
HWHM/FWHM/山谷間隔の誤差はそれぞれsigma_Gamma、2 sigma_Gamma、2 sigma_Gamma/sqrt(3)。
共分散行列が不安定な場合、推定できない誤差はNaNにして警告します。
定数背景で固定した傾きの誤差0は「推定した結果の0」ではなく「固定」を意味します。

## 出力

- `results/fit_results.csv`：1ファイル・1掃引区間につき1行。Excelでも開けるUTF-8 BOM付き。
- `results/diagnostics/*.png`：測定値、曲線、残差、誤差、警告。
- `results/covariance/*.csv`：物理パラメータ5個の共分散。行・列名に単位を記載。
- `results/run_summary.json`：条件、件数、重複座標など。
- `results/plots/*.png`：CSVだけから作る8枚の図。

CSVには日付、座標、掃引方向、点数、元ファイルのSHA256、各パラメータと標準誤差、
残差RMS、残差から推定したnoise_sigma_est、残差の隣接相関、成否と警告を記録します。
`n_raw`と`n_removed`は元ファイル全体、`n_points`は範囲制限後の各区間の点数です。

## 警告の読み方

`ok`は下記の自動判定に該当しなかったという意味で、モデルの正しさの保証ではありません。
`warning`も数値と誤差を残します。`failed`はその区間の解析失敗で、後続は処理します。
failedが1行でもあれば終了コード1です。入力全体が無い等は終了コード2です。

| warning | 意味 |
|---|---|
| weak_signal | フィット微分線の片側ピーク/残差標準偏差 < 5 |
| correlated_residuals | 取得順の残差の隣接相関の絶対値 > 0.3 |
| short_baseline_range | f0の左右のどちらかが2 Gamma未満の範囲しかない |
| derivative_extremum_outside_scan | 微分線の山または谷の理論位置が測定範囲外 |
| parameter_at_bound | 制約境界に近い推定 |
| width_below_sampling_step | Gammaが代表測定間隔未満 |
| large_parameter_uncertainty | Gammaの標準誤差がGammaの半分超、またはf0の標準誤差がGamma超 |
| ill_conditioned_covariance | ヤコビアンが不安定・ランク不足で誤差を算出できない |
| nonfinite_rows_removed | NaN/Inf行を除去した |
| multiple_sweep_segments | 周波数の折返しがあり、分割して解析した |
| plot_failed | 数値解析後の確認図作成に失敗した |

## CSVの図示

f0とFWHMについて、各zでのx依存・各xでのz依存を1 sigmaのエラーバーで表示します。
またxz平面に各値と標準誤差を色付きで表示します。未測定点は補間しません。
座標の単位は未確認なのでx,zのみ表示。丸はok、三角はwarningです。
誤差NaNの点にはエラーバーを描きません。failed行は除外します。
同日・同番号のファイルや複数の掃引区間が同一座標に残った場合は、空間図に件数 `n` を注記します。
warningを除く場合は `python plot_nmr_results.py --only-ok --output results/plots_only_ok`。

## 検証

`python -m unittest discover -s . -p "test_*.py"` で、既知のパラメータの模擬データからの
回復、誤差・共分散の変換、反転信号、掃引分割、失敗データの記録を確認できます。
