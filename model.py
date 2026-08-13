"""
Models and evaluation.

  zero      predict no move. R2_zero is 0 by definition.
  naive     sign(spot_imbalance), scaled to the median absolute move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import splits as S


# ------------------------------------------------------------------ models

def make_ridge(alpha: float = 1.0):
    """Linear baseline. Scaler inside the pipeline so it refits per fold."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_lgbm(**kw):
    """Gradient-boosted trees chosen low signal-to-noise target

      num_leaves=15, max_depth=4   shallow
      learning_rate=0.03 + 400 est slow shrinkage.
      min_child_samples=500        requires sufficient samples for a split
      subsample / colsample 0.8    decorrelates trees
      reg_lambda=1.0               L2, mirroring the ridge baseline.

    No early stopping against the validation fold. 
    """
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    params.update(kw)
    return LGBMRegressor(**params)


# -------------------------------------------------------------- benchmarks

def predict_zero(df, feats=None) -> np.ndarray:
    return np.zeros(len(df))


def predict_naive(df, feats=None, col: str = "spot_imbalance",
                  scale: float | None = None) -> np.ndarray:
    """sign(queue imbalance), scaled to a typical NON-ZERO move.

    The scale affects R2 and RMSE but not IC or directional accuracy.

    78% of this target is EXACTLY zero: want to condition on non-zero
    """
    if scale is None:
        nz = np.abs(df["y_bps"].to_numpy())
        nz = nz[nz > 0]
        scale = float(np.median(nz)) if len(nz) else 1.0
    return np.sign(df[col].to_numpy()) * scale


# ------------------------------------------------------------------- runner

def run_cv(df: pd.DataFrame, feats: list[str], label: str,
           model_fn=make_ridge, n_splits: int = 5, theta: float = 1.0,
           horizon_s: float = 5.0, embargo_s: float = 60.0,
           target: str = "y_bps", collect_preds: bool = False):
    """Purged walk-forward CV. Returns (per-fold DataFrame, out-of-sample preds).

    A fresh model per fold.
    """
    rows, preds = [], []
    for i, (tr, va) in enumerate(S.purged_walk_forward(
            df, n_splits=n_splits, horizon_s=horizon_s, embargo_s=embargo_s)):
        ytr = df.iloc[tr][target]
        yva = df.iloc[va][target]

        if model_fn in (predict_zero, predict_naive):
            p = model_fn(df.iloc[va])          # no fitting for benchmarks
        else:
            m = model_fn()
            m.fit(df.iloc[tr][feats], ytr)
            p = m.predict(df.iloc[va][feats])

        rows.append({"model": label, "fold": i,
                     **S.evaluate(yva, p, theta=theta)})
        if collect_preds:
            preds.append(pd.DataFrame({
                "t": df.iloc[va]["t"].to_numpy(),
                "y": yva.to_numpy(), "p": np.asarray(p), "fold": i}))

    cv = pd.DataFrame(rows)
    oos = pd.concat(preds, ignore_index=True) if collect_preds else None
    return cv, oos


def results_table(df: pd.DataFrame, feats: list[str], n_splits: int = 5,
                  theta: float = 1.0, include_lgbm: bool = True) -> pd.DataFrame:
    specs = [("zero", predict_zero), ("naive imbalance", predict_naive),
             ("ridge", make_ridge)]
    if include_lgbm:
        try:
            import lightgbm  # noqa: F401
            specs.append(("lightgbm", make_lgbm))
        except ImportError:
            print("lightgbm not installed — skipping (pip install lightgbm)")

    out = []
    for label, fn in specs:
        cv, _ = run_cv(df, feats, label, model_fn=fn,
                       n_splits=n_splits, theta=theta)
        g = cv.groupby("model").agg(
            r2_zero_mean=("r2_zero", "mean"), r2_zero_std=("r2_zero", "std"),
            ic_mean=("ic", "mean"), ic_std=("ic", "std"),
            dir_acc=(f"dir_acc_gt_{theta}bp", "mean"),
            rmse=("rmse_bps", "mean"))
        out.append(g)
    tbl = pd.concat(out)
    tbl["beats_naive_by"] = tbl.r2_zero_mean - tbl.loc["naive imbalance", "r2_zero_mean"]
    return tbl.round(5)


def verdict(tbl: pd.DataFrame, a: str = "ridge", b: str = "lightgbm") -> None:
    if b not in tbl.index or a not in tbl.index:
        return
    diff = tbl.loc[b, "r2_zero_mean"] - tbl.loc[a, "r2_zero_mean"]
    noise = max(tbl.loc[a, "r2_zero_std"], tbl.loc[b, "r2_zero_std"])
    print(f"\n{b} - {a} = {diff:+.5f} r2_zero;  fold sigma = {noise:.5f}")
    if diff > noise:
        print(f"-> {b} improves by more than one fold sigma. Worth keeping.")
    elif diff > 0:
        print(f"-> improvement is INSIDE fold noise. Not evidence {b} is better; "
              f"prefer the simpler model.")
    else:
        print(f"-> {b} is no better. Keep {a}.")



def make_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    