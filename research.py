"""
Helpers for loading the curated capture and building mid-price labels.

Import from a notebook:  import research as R

The curated tree is Hive-partitioned:
    curated/venue=<spot|futures>/symbol=<SYM>/stream=<kind>/date=/hour=/

Stream kinds and their key columns (every row also has recv_ns, sess, seq):

  book_ticker   bid_px bid_qty ask_px ask_qty update_id exch_event_ns exch_txn_ns
                -> best bid/offer. THE stream for mid-price.
                -> spot has NO exchange timestamp (null); use recv_ns.
  agg_trade     px qty agg_id first_trade_id last_trade_id buyer_is_maker
                exch_event_ns exch_trade_ns
                -> buyer_is_maker=True means the aggressor was a SELLER.
  depth         bids asks (list<struct<px,qty>>) first_update_id final_update_id
                prev_final_update_id exch_event_ns exch_txn_ns
                -> incremental diffs, NOT a book. Must be applied to a snapshot.
  mark_price    mark_px index_px settle_px funding_rate next_funding_ns  (futures)
  force_order   side order_type orig_qty px avg_px status ...            (futures)

Timestamps are int64 UTC nanoseconds. recv_ns is local receive; exch_event_ns is
the exchange's own clock. They differ by network latency (~140ms London->Tokyo),
which is 3% of a 5-second horizon — so which one you build labels on matters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

NS = 1_000_000_000

DEFAULT_COLS = {
    "book_ticker": ["recv_ns", "exch_event_ns", "exch_txn_ns", "sess", "update_id",
                    "bid_px", "bid_qty", "ask_px", "ask_qty"],
    "agg_trade": ["recv_ns", "exch_event_ns", "exch_trade_ns", "sess",
                  "px", "qty", "buyer_is_maker"],
    "mark_price": ["recv_ns", "exch_event_ns", "sess", "mark_px", "index_px",
                   "funding_rate"],
    "force_order": ["recv_ns", "exch_event_ns", "sess", "side", "orig_qty",
                    "px", "avg_px"],
    "depth": None,   # nested; take everything
}


def path_for(root, venue: str, symbol: str, stream: str) -> Path:
    return (Path(root) / "curated" / f"venue={venue}" / f"symbol={symbol}"
            / f"stream={stream}")


def inventory(root) -> pd.DataFrame:
    """What has been captured, by venue/stream/date/hour. Cheap — metadata only."""
    rows = []
    for p in (Path(root) / "curated").rglob("*.parquet"):
        parts = dict(seg.split("=", 1) for seg in p.parts if "=" in seg)
        parts["rows"] = ds.dataset(p, format="parquet").count_rows()
        parts["mb"] = round(p.stat().st_size / 1e6, 1)
        rows.append(parts)
    if not rows:
        raise FileNotFoundError(f"no parquet under {Path(root)/'curated'} — "
                                "run to_parquet.py first")
    df = pd.DataFrame(rows)
    return df.sort_values(["venue", "stream", "date", "hour"]).reset_index(drop=True)


def load(root, venue: str, symbol: str = "BTCUSDT", stream: str = "book_ticker",
         hours: list[str] | None = None, dates: list[str] | None = None,
         last_hours: int | None = None,
         columns: list[str] | None = "default") -> pd.DataFrame:
    """Load one stream into a DataFrame, sorted by recv_ns.

    Filtering happens BEFORE reading, which is the difference between loading
    200MB and loading 4GB. Three ways to filter:

      last_hours=6            the 6 most recent CONTIGUOUS hours  <- usually this
      dates=["2026-08-10"]    whole days
      hours=["20","21"]       hour-of-day, applied ACROSS EVERY DATE

    `hours` is a partition-value filter, not a time range: hours=["20","21"]
    on a five-day capture returns hour 20 and 21 from all five days, with
    ~22-hour jumps between the blocks. That silently breaks anything computing
    returns or forward labels across the joins. Use last_hours for a
    contiguous window, or pass dates alongside hours to pin the day.
    """
    base = path_for(root, venue, symbol, stream)
    d = ds.dataset(base, format="parquet", partitioning="hive")

    # Hive partitioning infers types: date stays a string, but hour ("04")
    # infers as int32. Filter values must match or pyarrow raises on binding.
    def _cast(field_name, value):
        t = d.schema.field(field_name).type
        return int(value) if pa.types.is_integer(t) else str(value)

    flt = None
    if last_hours is not None:
        pairs = sorted(
            {(p.parent.parent.name.split("=", 1)[1], p.parent.name.split("=", 1)[1])
             for p in base.rglob("*.parquet")},
            key=lambda x: (x[0], int(x[1])),
        )[-last_hours:]
        if not pairs:
            raise FileNotFoundError(f"no parquet under {base}")
        for dt, hr in pairs:
            f = ((ds.field("date") == _cast("date", dt))
                 & (ds.field("hour") == _cast("hour", hr)))
            flt = f if flt is None else (flt | f)
    else:
        if dates is not None:
            flt = ds.field("date").isin([_cast("date", x) for x in dates])
        if hours is not None:
            h = ds.field("hour").isin([_cast("hour", x) for x in hours])
            flt = h if flt is None else (flt & h)

    if columns == "default":
        columns = DEFAULT_COLS.get(stream)
    if columns is not None:
        columns = [c for c in columns if c in d.schema.names]

    df = d.to_table(filter=flt, columns=columns).to_pandas()
    return df.sort_values("recv_ns", ignore_index=True)


def contiguous_blocks(df: pd.DataFrame, time_col: str = "recv_ns",
                      max_gap_s: float = 5.0) -> pd.Series:
    """Label rows with a block id that increments at every gap.

    Use this before any diff/pct_change/rolling operation. A return computed
    across a capture gap is not a return — it is the price change over however
    many hours were missing, and a handful of those will dominate a correlation.
    """
    ts = df[time_col].to_numpy(dtype="int64")
    return pd.Series(np.concatenate([[0], np.diff(ts) > max_gap_s * NS]).cumsum(),
                     index=df.index, name="block")



def add_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Mid, spread, size-weighted mid, and queue imbalance."""
    df = df.copy()
    df["mid"] = (df.bid_px + df.ask_px) / 2
    df["spread"] = df.ask_px - df.bid_px
    tot = df.bid_qty + df.ask_qty
    # weighted toward the price on the side with LESS size, i.e. the side more
    # likely to be consumed next
    df["weighted_mid"] = np.where(
        tot > 0, (df.ask_px * df.bid_qty + df.bid_px * df.ask_qty) / tot, df["mid"])
    df["imbalance"] = np.where(tot > 0, (df.bid_qty - df.ask_qty) / tot, 0.0)
    return df


def forward_price(df: pd.DataFrame, horizon_s: float = 5.0,
                  time_col: str = "recv_ns", price_col: str = "mid",
                  max_staleness_s: float = 1.0,
                  respect_sessions: bool = True) -> pd.DataFrame:
    """Price prevailing `horizon_s` later, plus the forward return.

    Two guards that matter, and are easy to forget:

    1. STALENESS. Across a gap, the last-known price before the gap would
       otherwise be carried forward and look like a real observation. Any
       lookup whose source is more than `max_staleness_s` old becomes NaN.
    2. SESSIONS. A label must not straddle a reconnect — the data either side
       is fine, but the interval between them is unobserved.

    Rows near the end of the sample get NaN because the future isn't there yet.
    Drop them; never fill them.
    """
    df = df.sort_values(time_col, ignore_index=True)
    ts = df[time_col].to_numpy(dtype="int64")
    px = df[price_col].to_numpy(dtype="float64")

    target = ts + int(horizon_s * NS)
    idx = np.searchsorted(ts, target, side="right") - 1

    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    ok &= (target - ts[safe]) <= int(max_staleness_s * NS)

    if respect_sessions and "sess" in df.columns:
        sess = df["sess"].to_numpy()
        ok &= sess == sess[safe]

    fwd = np.where(ok, px[safe], np.nan)
    out = df.copy()
    out[f"fwd_{price_col}_{horizon_s:g}s"] = fwd
    out[f"fwd_ret_{horizon_s:g}s"] = fwd / px - 1.0
    out[f"fwd_bps_{horizon_s:g}s"] = (fwd / px - 1.0) * 1e4
    return out


def to_datetime(df: pd.DataFrame, col: str = "recv_ns") -> pd.Series:
    return pd.to_datetime(df[col], unit="ns", utc=True)


def resample_last(df: pd.DataFrame, freq: str = "1s", time_col: str = "recv_ns",
                  cols: list[str] | None = None) -> pd.DataFrame:
    """Last observation on a fixed clock grid.

    Convenient, but note what it discards: update RATE is itself informative
    (it spikes with volatility), and a fixed grid throws that away. Prefer
    working in event time where you can.
    """
    g = df.set_index(pd.to_datetime(df[time_col], unit="ns", utc=True))
    if cols is not None:
        g = g[cols]
    return g.resample(freq).last().dropna(how="all")
