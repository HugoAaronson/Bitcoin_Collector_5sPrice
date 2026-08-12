"""
Feature construction on the 1-second grid.

A feature at grid point t may depend only on
records with recv_ns <= t. Every function here enforces that structurally rather
than by convention:

  * book state    -> merge_asof(direction="backward") on recv_ns
  * trade flow    -> trades binned into (t-1s, t], then a trailing rolling sum
  * volatility    -> rolling std of returns already realised at t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = 1_000_000_000


# ---------------------------------------------------------------- book joins

def asof_book(grid: pd.DataFrame, book: pd.DataFrame, prefix: str,
              max_age_s: float = 5.0) -> pd.DataFrame:
    """Attach the last known book state at or before each grid point.

    Adds <prefix>_mid, _spread, _imbalance, _wmid_dev and <prefix>_age_s.
    Rows older than max_age_s get NaN features (not dropped — the caller
    decides).

    _wmid_dev is the size-weighted mid MINUS the mid, in basis points.
    """
    b = book.sort_values("recv_ns", ignore_index=True).copy()
    if "mid" not in b:
        b["mid"] = (b.bid_px + b.ask_px) / 2
    tot = b.bid_qty + b.ask_qty
    b["_imb"] = np.where(tot > 0, (b.bid_qty - b.ask_qty) / tot, 0.0)
    wmid = np.where(tot > 0,
                    (b.ask_px * b.bid_qty + b.bid_px * b.ask_qty) / tot, b["mid"])
    b["_wdev"] = (wmid - b["mid"]) / b["mid"] * 1e4
    b["_spr"] = (b.ask_px - b.bid_px) / b["mid"] * 1e4

    b["_src_ns"] = b["recv_ns"]
    cols = ["recv_ns", "mid", "_spr", "_imb", "_wdev", "_src_ns"]
    merged = pd.merge_asof(
        grid[["t_ns"]].sort_values("t_ns"),
        b[cols].rename(columns={"recv_ns": "t_ns"}),
        on="t_ns", direction="backward",
    )

    age_s = (merged.t_ns - merged._src_ns) / NS
    out = pd.DataFrame({f"{prefix}_age_s": age_s.to_numpy()})
    stale = out[f"{prefix}_age_s"].isna() | (out[f"{prefix}_age_s"] > max_age_s)
    for src, dst in [("mid", "mid"), ("_spr", "spread_bps"),
                     ("_imb", "imbalance"), ("_wdev", "wmid_dev_bps")]:
        out[f"{prefix}_{dst}"] = np.where(stale, np.nan, merged[src].to_numpy())
    return out


# --------------------------------------------------------------- trade flow

def trade_flow(grid: pd.DataFrame, trades: pd.DataFrame, prefix: str,
               windows_s: tuple[int, ...] = (1, 5, 60),
               freq_s: float = 1.0) -> pd.DataFrame:
    """Signed volume, trade count and volume over trailing windows.

    Binning: a trade at time tau belongs to the bin labelled ceil(tau / step),
    i.e. the half-open interval (t - 1s, t]. A trailing rolling sum of N bins
    ending at t therefore covers exactly (t - N seconds, t] and uses nothing
    after t.

    SIGN CONVENTION: Binance's `m` / buyer_is_maker is True when the resting
    BUY order was hit, i.e. the AGGRESSOR WAS A SELLER. So buyer_is_maker=True
    contributes negative signed flow. This can be inverted.
    """
    step = int(freq_s * NS)
    t = trades.sort_values("recv_ns")
    tau = t.recv_ns.to_numpy(dtype="int64")
    bin_ns = -((-tau) // step) * step          # ceil division -> (t-1s, t]

    qty = t.qty.to_numpy(dtype="float64")
    signed = np.where(t.buyer_is_maker.to_numpy(), -qty, qty)

    per_bin = pd.DataFrame({"t_ns": bin_ns, "signed": signed,
                            "qty": qty, "n": 1.0}).groupby("t_ns").sum()

    # reindex onto the COMPLETE grid so a rolling window of N rows is exactly
    # N seconds; empty seconds are genuinely zero trades, not missing data
    per_bin = per_bin.reindex(grid.t_ns.to_numpy(), fill_value=0.0)

    out = pd.DataFrame(index=per_bin.index)
    for w in windows_s:
        k = int(w / freq_s)
        sv = per_bin["signed"].rolling(k, min_periods=1).sum()
        vol = per_bin["qty"].rolling(k, min_periods=1).sum()
        out[f"{prefix}_signed_vol_{w}s"] = sv.to_numpy()
        out[f"{prefix}_n_trades_{w}s"] = per_bin["n"].rolling(k, min_periods=1).sum().to_numpy()
        # normalised order-flow imbalance: bounded, comparable across regimes
        out[f"{prefix}_ofi_{w}s"] = np.where(vol > 0, sv / vol, 0.0)
    return out.reset_index(drop=True)


# --------------------------------------------------------------- volatility

def realised_vol(grid: pd.DataFrame, windows_s: tuple[int, ...] = (10, 60),
                 price_col: str = "mid", freq_s: float = 1.0) -> pd.DataFrame:
    """Rolling std of 1-second log returns, in bps.

    At t the return r(t) = ln(m(t)/m(t-1)) is already realised, so a trailing
    window ending at t is causal. Uses the grid's own mid, so it inherits the
    grid's gap handling.
    """
    m = grid[price_col].to_numpy(dtype="float64")
    r = np.full(len(m), np.nan)
    r[1:] = np.log(m[1:] / m[:-1]) * 1e4
    s = pd.Series(r)
    out = pd.DataFrame()
    for w in windows_s:
        k = int(w / freq_s)
        out[f"rv_{w}s_bps"] = s.rolling(k, min_periods=max(3, k // 3)).std().to_numpy()
    out["ret_1s_bps"] = r
    return out


# ------------------------------------------------------------------ assemble

def build(grid: pd.DataFrame, spot_book: pd.DataFrame, fut_book: pd.DataFrame,
          spot_trades: pd.DataFrame, fut_trades: pd.DataFrame,
          mark: pd.DataFrame | None = None,
          max_age_s: float = 5.0) -> pd.DataFrame:
    """Assemble the full causal feature frame aligned to the grid."""
    parts = [
        grid.reset_index(drop=True),
        asof_book(grid, spot_book, "spot", max_age_s),
        asof_book(grid, fut_book, "fut", max_age_s),
        trade_flow(grid, spot_trades, "spot"),
        trade_flow(grid, fut_trades, "fut"),
        realised_vol(grid),
    ]
    X = pd.concat(parts, axis=1)

    # spot-perp basis: the mechanical link between the two instruments
    X["basis_bps"] = (X.fut_mid / X.spot_mid - 1) * 1e4

    if mark is not None and len(mark):
        mk = pd.merge_asof(
            grid[["t_ns"]].sort_values("t_ns"),
            mark.sort_values("recv_ns")[["recv_ns", "funding_rate"]]
                .rename(columns={"recv_ns": "t_ns"}),
            on="t_ns", direction="backward")
        X["funding_rate"] = mk.funding_rate.to_numpy()
    return X


FEATURES = [
    "spot_imbalance", "spot_spread_bps", "spot_wmid_dev_bps",
    "fut_imbalance", "fut_spread_bps", "fut_wmid_dev_bps",
    "basis_bps",
    "spot_ofi_1s", "spot_ofi_5s", "fut_ofi_1s", "fut_ofi_5s", "fut_ofi_60s",
    "fut_n_trades_5s", "spot_n_trades_5s",
    "rv_10s_bps", "rv_60s_bps", "ret_1s_bps",
]


def assert_causal(X: pd.DataFrame, prefixes=("spot", "fut")) -> None:
    """Structural guarantee: no joined observation post-dates its grid point.

    Every book join records the age of the observation it
    used, and a negative age means data from the future got in.
    """
    bad = []
    for p in prefixes:
        col = f"{p}_age_s"
        if col in X:
            neg = (X[col] < 0).sum()
            if neg:
                bad.append(f"{col}: {neg} rows with negative age")
    if bad:
        raise AssertionError("LOOKAHEAD: " + "; ".join(bad))
    print("causality OK — every joined observation is at or before its grid point")


def ic_profile(X: pd.DataFrame, features=None, target: str = "y_bps",
               shifts=(-3, -2, -1, 0, 1, 2, 3)) -> pd.DataFrame:
    #Mean |IC| with features shifted by k grid steps.
    features = features or FEATURES
    features = [f for f in features if f in X.columns]
    d = X.dropna(subset=features + [target])
    n = len(d)
    rows = []
    for k in shifts:
        sh = d[features].shift(k)
        j = sh.join(d[target]).dropna()
        rows.append({"shift": k,
                     "mean_abs_IC": j[features].corrwith(j[target]).abs().mean(),
                     "n": len(j)})
    out = pd.DataFrame(rows).set_index("shift")
    band = 1 / np.sqrt(max(n, 1))
    print(f"n = {n:,}   noise band ~ {band:.4f}   "
          "(differences smaller than this mean nothing)")
    print(out.round(5).to_string())
    if out.loc[0, "mean_abs_IC"] < 2 * band:
        print("NOTE: signal at shift 0 is within noise — this profile cannot "
              "diagnose a leak. Rely on assert_causal instead.")
    return out

