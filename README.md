# BTCUSDT 5-Second Mid-Price Forecast

## 1. What?

This project is for collecting live BTC market data (both spot and futures) to attempt to predict the 5-second spot mid-price move.

---

## 2. How to run it

**Required:**

```powershell
python -m venv .venv
pip install -r requirements.txt
```

**Collection of raw data**

- Files: `collector.py`
- Run in powershell:

```powershell
.venv\Scripts\python.exe collector.py --root .\data --depth
```

**Converting it to Parquet**

- Files: `to_parquet.py`
- Run in powershell:

```powershell
.venv\Scripts\python.exe to_parquet.py --root .\data
```

- Run in powershell if you want to redo all conversion:

```powershell
.venv\Scripts\python.exe to_parquet.py --root .\data --force
```

**Audit**

- Files: `check.py`
- Run in powershell:

```powershell
.venv\Scripts\python.exe check.py --root .\data
```

**Research:** Run the following notebook with the corresponding python files

- `Analysis_Predictor.IPYNB`
- `features.py`
- `model.py`
- `research.py`
- `splits.py`

---

## 3. What I used

ASUS ROG Zephyrus, Windows 11, mains power, domestic UK broadband, Windows Update paused and sleep/hibernate/display-blanking disabled

Windows Modern Standby disconnected collector which meant I had to prevent screen turning off

---

## 4. Data quality summary

Final run of `check.py` is available at [`Final_Collection_check`](Final_Collection_check)

| | |
|---|---|
| Window | 2026-08-06 20:14 to 2026-08-11 15:57 UTC |
| time-span | 115.7 hrs |
| Covered data | around 109.6 h (95% of time-span) |
| Total records | around 40M |
| Instruments | BTCUSDT spot, BTCUSDT USDⓈ-M perpetual |
| Streams | `@bookTicker`, `@aggTrade`, `@depth@100ms` (both venues); `@markPrice@1s`, `@forceOrder` (futures) |
| Storage | Parquet, Hive-partitioned by venue/symbol/stream/date/hour, zstd |

| When (UTC) | Duration | Connections | Cause |
|---|---|---|---|
| 08-06 20:16 | 187 s | spot, fut_public | OS suspend (Windows Modern Standby, S0ix) |
| 08-06 20:27 | 611–636 s | spot, fut_public | OS suspend; same root cause |
| 08-07 16:40 | 3 h 43 m | all three | Deliberate stop; travel between locations |
| 08-07 20:33 | 2 h 06 m | all three | Host powered off; restarted manually |
| 08-08 18:21 | 3–37 s | all three | Transient network drop, auto-recovered |
| 08-08 19:34 | 2–12 s | fut_market | Transient network drop, auto-recovered |
| 08-09 08:18–09:33 | 2–9 s ×3 | fut_market | Transient network drops, auto-recovered |
| 08-09 22:45 | 12 s | spot | Transient network drop, auto-recovered |
| 08-11 09:34 | 4–8 s ×3 | spot | Transient network drops, auto-recovered |

Most data loss was from known down-time (travel related)

97.5% of lost time is from these. Under 90 s across the rest.

- constant-rate controls: `markPrice@1s` = exactly 3600/hour in every complete hour; `depth@100ms` at 35.29k–35.99k of 36k
- latency p50 ~125 ms, p90 142–188 ms, max comes from reconnections not actual latency
- Modelled the final 80 h to avoid the large gaps
- dropped 198 unusable points (0.07%, mostly single points in quiet books)

---

## 5. Method

**target:** spot mid, 1 s grid, `y = 1e4 · ln(m(t+5s)/m(t))`

**information boundary** on receive time, and the neat consequence that spot `bookTicker` has no exchange timestamp so this was forced anyway

### features

**Baseline Features:** Chose 4 different groups of Baseline features:

**Book:**

- `spot_imbalance`, `fut_imbalance`: These correspond to relationship between bid and ask quantities
- `spot_spread_bps`, `fut_spread_bps`: spreads between ask and bid
- `spot_wmid_dev_bps`, `fut_wmid_dev_bps`: mid weighted by size of positions minus mid

**between venue:**

- `basis_bps`: relationship between the mid-price of futures and spot

**Trade flow:**

- `_ofi_1s`, `_ofi_5s`, `fut_ofi_60s`: signed flow normalised by volume over different timescales (implicitly gives momentum features for volume)

**Volume/momentum:**

- `rv_10s_bps`, `rv_60s_bps`: rolling standard deviations (longer timescales as volatility is slower-varying)
- `ret_1s_bps`: previous 1 second log-returns

### validation

- frozen 20% holdout
- purged walk-forward Cross-validation
- 60s embargo between each fold given that is length of the effect of longest feature

---

## 6. Headline numbers

Holdout, evaluated once: **R²_zero 0.079, IC 0.287, Spearman 0.425, Directional Accuracy 0.801** at threshhold `|y|>θ=1 bp` (n=5,231, 9.1% of sample), RMSE 0.673 bps. DA is flat 0.80–0.82 from θ=0.1 to θ=2.

| theta_bps | n | pct_of_sample | dir_acc |
|---|---|---|---|
| 0.10 | 13702 | 0.2385 | 0.8170 |
| 0.25 | 11515 | 0.2004 | 0.8129 |
| 0.50 | 8791 | 0.1530 | 0.8042 |
| 1.00 | 5231 | 0.0910 | 0.8006 |
| 2.00 | 1806 | 0.0314 | 0.7957 |
| 5.00 | 89 | 0.0015 | 0.6854 |

zero-benchmark r2_zero: 0.0

**Model comparison** (5-fold purged walk-forward CV on the development set (not on holdout)):

| model | r2_zero_mean | r2_zero_std | ic_mean | ic_std | dir_acc | rmse | beats_naive_by |
|---|---|---|---|---|---|---|---|
| zero | 0.00000 | 0.00000 | NaN | NaN | 0.00000 | 0.56837 | 0.51569 |
| naive imbalance | -0.51569 | 0.40073 | 0.18561 | 0.06272 | 0.79235 | 0.66298 | 0.00000 |
| ridge | 0.04463 | 0.02365 | 0.22157 | 0.05302 | 0.79352 | 0.55384 | 0.56032 |
| lightgbm | 0.06400 | 0.03626 | 0.26348 | 0.06300 | 0.76869 | 0.54659 | 0.57969 |

---

## 7. What I found

- direction is almost entirely controlled by the sign in imbalance: naive DA 0.792 vs ridge 0.794
- ridge's contribution is calibrating to much smaller magnitudes to account for 78% being no returns: R²_zero naive had −0.516 and ridge was 0.045 as a result of similar DA but ridge had a far smaller magnitude of std (0.16 vs 0.7).
- signal is weaker for quiet periods (0.014) than medium or busier ones (0.065/0.052)
- LightGBM provided little improvement which meant that we kept the ridge
- vast majority (78%) of all the labels are 0.

**Stability across folds:**

| fold | r2_zero | ic | dir_acc_gt_1.0bp | n |
|---|---|---|---|---|
| 0 | 0.0160 | 0.1552 | 0.7725 | 38288 |
| 1 | 0.0348 | 0.1844 | 0.8325 | 38288 |
| 2 | 0.0375 | 0.2288 | 0.7942 | 38288 |
| 3 | 0.0781 | 0.2893 | 0.8079 | 38288 |
| 4 | 0.0567 | 0.2501 | 0.7605 | 38291 |

**stability across days:**

| t | r2_zero | ic | n |
|---|---|---|---|
| 2026-08-08 | 0.0034 | 0.1366 | 19099.0 |
| 2026-08-09 | 0.0266 | 0.1632 | 86114.0 |
| 2026-08-10 | 0.0605 | 0.2605 | 86230.0 |

**stability across regimes:**

| regime | r2_zero | ic | n |
|---|---|---|---|
| quiet | 0.0142 | 0.1389 | 63815.0 |
| mid | 0.0646 | 0.2672 | 63819.0 |
| busy | 0.0524 | 0.2393 | 63809.0 |

The first two show that more data improves the score, the last shows that quiet regimes have less signal.

---

## 8. Validation

- When the target was shuffled breaking connection between features and target `r2_zero_mean: −0.00019`
- zero-benchmark (only outputs zero on all inputs) returns exactly 0.0
- structural causality assertion called on every join
- holdout evaluated once

---

## 10. What I'd do with another week

1. Reconstructing the book for more detailed features
2. Capture data from second venue potentially leading to arbitrage opportunities
3. Compare prices here to competing cryptocurrencies such as Ethereum
4. Longer collection time if it meant that I could compare the same days of the week during different weeks (more likely to be periodic over 7 days)
5. model the sign and magnitude separately, binary label for whether it is 0
6. explore the effect of quiet vs busy regimes further

---

## 11. What I'd change about what I did

- run on a VPS not a laptop
- use chrony rather than Windows Time
- budget more time for analysis and feature exploration

---

## 12. Known limitations

- Spot BBO has no exchange timestamp
- Windows Time is loose
- 4 market days isn't enough to capture range of possible regimes.
- Millisecond exchange make it harder to align across streams

---

## 13. Economics

If the return has a standard deviation of 0.7bp, this is completely swallowed by any fee especially if it is at least 5bp. Potentially useful for skewing quotes for market makers but still too small to be significant.
