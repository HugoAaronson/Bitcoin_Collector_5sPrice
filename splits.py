"""
Time-series splitting: frozen holdout, purged walk-forward CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = 1_000_000_000


def frozen_holdout(df: pd.DataFrame, frac: float = 0.20,
                   time_col: str = "t_ns", horizon_s: float = 5.0,
                   embargo_s: float = 60.0, by: str = "rows"):
    d = df.sort_values(time_col, ignore_index=True)
    t = d[time_col].to_numpy(dtype="int64")

    cut_rows = int(t[int(len(d) * (1 - frac))])
    cut_time = int(t[0] + (t[-1] - t[0]) * (1 - frac))
    cut_ns = cut_rows if by == "rows" else cut_time

    holdout = d[d[time_col] >= cut_ns]
    limit = cut_ns - int((horizon_s + embargo_s) * NS)
    develop = d[d[time_col] <= limit]

    fmt = lambda x: pd.to_datetime(x, unit="ns", utc=True)
    print(f"frozen holdout, by={by}, at {fmt(cut_ns)}")
    if cut_rows != cut_time:
        print(f"  (row-based cut {fmt(cut_rows)}, time-based cut {fmt(cut_time)}, "
              f"differ by {abs(cut_rows-cut_time)/NS/60:.1f} min)")
    print(f"  develop: {len(develop):,} rows   holdout: {len(holdout):,} rows "
          f"({len(holdout)/len(d):.1%} of samples)")
    print(f"  purged+embargoed at the boundary: "
          f"{len(d)-len(develop)-len(holdout):,} rows")
    return develop.reset_index(drop=True), holdout.reset_index(drop=True)


def purged_walk_forward(df: pd.DataFrame, n_splits: int = 5,
                        time_col: str = "t_ns", horizon_s: float = 5.0,
                        embargo_s: float = 60.0, expanding: bool = True):
    """Yield (train_idx, val_idx) for walk-forward CV with purge + embargo.

    `expanding=True` grows the training window each fold; False keeps it a
    fixed rolling length.

    GUARANTEE, asserted below: for every fold,
        max(train t) + horizon_s + embargo_s <= min(val t)
    """
    d = df.sort_values(time_col, ignore_index=True)
    t = d[time_col].to_numpy(dtype="int64")
    n = len(d)

    fold_size = n // (n_splits + 1)
    guard = int((horizon_s + embargo_s) * NS)

    for k in range(n_splits):
        val_start_pos = fold_size * (k + 1)
        val_end_pos = fold_size * (k + 2) if k < n_splits - 1 else n
        val_idx = np.arange(val_start_pos, val_end_pos)
        if len(val_idx) == 0:
            continue

        v_start_ns = t[val_idx[0]]
        # purge + embargo: training labels must fully resolve before validation
        train_mask = t <= (v_start_ns - guard)
        train_idx = np.flatnonzero(train_mask)

        if not expanding:
            keep = fold_size
            train_idx = train_idx[-keep:] if len(train_idx) > keep else train_idx

        if len(train_idx) == 0:
            continue

        assert t[train_idx].max() + guard <= t[val_idx].min(), \
            "purge/embargo violated"
        yield train_idx, val_idx


def describe_splits(df: pd.DataFrame, n_splits: int = 5, **kw) -> pd.DataFrame:
    """Table of fold sizes and boundary times"""
    rows = []
    for i, (tr, va) in enumerate(purged_walk_forward(df, n_splits, **kw)):
        t = df.sort_values(kw.get("time_col", "t_ns"))[kw.get("time_col", "t_ns")].to_numpy()
        rows.append({
            "fold": i,
            "n_train": len(tr), "n_val": len(va),
            "train_end": pd.to_datetime(t[tr].max(), unit="ns", utc=True),
            "val_start": pd.to_datetime(t[va].min(), unit="ns", utc=True),
            "gap_s": (t[va].min() - t[tr].max()) / NS,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- metrics

def evaluate(y_true, y_pred, theta: float = 1.0) -> dict:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    sse = np.sum((y_true - y_pred) ** 2)
    r2_zero = 1 - sse / np.sum(y_true ** 2)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan
    ic_s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

    big = np.abs(y_true) > theta
    da = float(np.mean(np.sign(y_pred[big]) == np.sign(y_true[big]))) if big.sum() else np.nan

    return {
        "r2_zero": r2_zero,
        "ic": ic,
        "ic_spearman": ic_s,
        f"dir_acc_gt_{theta}bp": da,
        f"n_gt_{theta}bp": int(big.sum()),
        "rmse_bps": float(np.sqrt(sse / len(y_true))),
        "n": len(y_true),
    }


def theta_sensitivity(y_true, y_pred, thetas=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rows = []
    for th in thetas:
        big = np.abs(y_true) > th
        rows.append({
            "theta_bps": th,
            "n": int(big.sum()),
            "pct_of_sample": big.mean(),
            "dir_acc": float(np.mean(np.sign(y_pred[big]) == np.sign(y_true[big])))
                       if big.sum() else np.nan,
        })
    return pd.DataFrame(rows)
