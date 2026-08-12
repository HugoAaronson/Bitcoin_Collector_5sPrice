#!/usr/bin/env python3
"""
Convert completed raw JSONL.gz hours into typed, partitioned Parquet.

Runs as a SEPARATE process from the collector, on completed hours only. If this
crashes, the raw capture is untouched and can just be re-run.

Output layout (Hive-style, so pyarrow/duckdb/polars can read it as one dataset):
    <root>/curated/venue=<spot|futures>/symbol=<SYM>/stream=<kind>/
           date=YYYY-MM-DD/hour=HH/part-<conn>.parquet

Every row carries BOTH timestamps, normalised to int64 UTC nanoseconds:
    recv_ns        our local receive time  (time.time_ns)
    exch_event_ns  the exchange's own event time

Unit note: the spot connection requests timeUnit=MICROSECOND, futures returns
milliseconds. Both are normalised to nanoseconds here so downstream code never
has to care. Spot @bookTicker has no exchange timestamp at all: that column is
null for those rows by design, not by bug. Use update_id for spot BBO ordering.
"""
from __future__ import annotations
 
import argparse
import gzip
import json
import logging
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
 
import pyarrow as pa
import pyarrow.parquet as pq
 
log = logging.getLogger("to_parquet")
 
# Exchange timestamp unit -> nanosecond multiplier, per connection.
TS_MULT = {"spot": 1_000, "fut_public": 1_000_000, "fut_market": 1_000_000}
VENUE = {"spot": "spot", "fut_public": "futures", "fut_market": "futures"}
 
LEVEL = pa.list_(pa.struct([("px", pa.float64()), ("qty", pa.float64())]))
 
COMMON = [
    ("recv_ns", pa.int64()),
    ("mono_ns", pa.int64()),
    ("sess", pa.string()),
    ("seq", pa.int64()),
]
 
SCHEMAS = {
    "book_ticker": pa.schema(COMMON + [
        ("exch_event_ns", pa.int64()),   # null on spot
        ("exch_txn_ns", pa.int64()),     # null on spot
        ("update_id", pa.int64()),
        ("bid_px", pa.float64()),
        ("bid_qty", pa.float64()),
        ("ask_px", pa.float64()),
        ("ask_qty", pa.float64()),
    ]),
    "agg_trade": pa.schema(COMMON + [
        ("exch_event_ns", pa.int64()),
        ("exch_trade_ns", pa.int64()),
        ("agg_id", pa.int64()),
        ("px", pa.float64()),
        ("qty", pa.float64()),
        ("first_trade_id", pa.int64()),
        ("last_trade_id", pa.int64()),
        ("buyer_is_maker", pa.bool_()),
    ]),
    "depth": pa.schema(COMMON + [
        ("exch_event_ns", pa.int64()),
        ("exch_txn_ns", pa.int64()),     # futures only
        ("first_update_id", pa.int64()),
        ("final_update_id", pa.int64()),
        ("prev_final_update_id", pa.int64()),  # futures 'pu', null on spot
        ("bids", LEVEL),
        ("asks", LEVEL),
    ]),
    "mark_price": pa.schema(COMMON + [
        ("exch_event_ns", pa.int64()),
        ("mark_px", pa.float64()),
        ("index_px", pa.float64()),
        ("settle_px", pa.float64()),
        ("funding_rate", pa.float64()),
        ("next_funding_ns", pa.int64()),
    ]),
    "force_order": pa.schema(COMMON + [
        ("exch_event_ns", pa.int64()),
        ("exch_trade_ns", pa.int64()),
        ("side", pa.string()),
        ("order_type", pa.string()),
        ("time_in_force", pa.string()),
        ("orig_qty", pa.float64()),
        ("px", pa.float64()),
        ("avg_px", pa.float64()),
        ("status", pa.string()),
        ("last_filled_qty", pa.float64()),
        ("filled_accum_qty", pa.float64()),
    ]),
}
 
 
def classify(stream: str) -> tuple[str, str]:
    """stream name -> (SYMBOL, kind)"""
    sym, _, rest = stream.partition("@")
    sym = sym.upper()
    if rest.startswith("bookTicker"):
        return sym, "book_ticker"
    if rest.startswith("aggTrade"):
        return sym, "agg_trade"
    if rest.startswith("depth"):
        return sym, "depth"
    if rest.startswith("markPrice"):
        return sym, "mark_price"
    if rest.startswith("forceOrder"):
        return sym, "force_order"
    return sym, "other"
 
 
def _f(v):
    return float(v) if v is not None else None
 
 
def _ns(v, mult):
    return int(v) * mult if v is not None else None
 
 
def _levels(raw):
    return [{"px": float(p), "qty": float(q)} for p, q in (raw or [])]
 
 
def normalise(kind: str, rec: dict, mult: int) -> dict | None:
    d = rec["d"]
    base = {
        "recv_ns": rec["recv_ns"],
        "mono_ns": rec.get("mono_ns"),
        "sess": rec["sess"],
        "seq": rec["seq"],
    }
    if kind == "book_ticker":
        return base | {
            "exch_event_ns": _ns(d.get("E"), mult),
            "exch_txn_ns": _ns(d.get("T"), mult),
            "update_id": int(d["u"]),
            "bid_px": _f(d["b"]), "bid_qty": _f(d["B"]),
            "ask_px": _f(d["a"]), "ask_qty": _f(d["A"]),
        }
    if kind == "agg_trade":
        return base | {
            "exch_event_ns": _ns(d.get("E"), mult),
            "exch_trade_ns": _ns(d.get("T"), mult),
            "agg_id": int(d["a"]),
            "px": _f(d["p"]), "qty": _f(d["q"]),
            "first_trade_id": int(d["f"]), "last_trade_id": int(d["l"]),
            "buyer_is_maker": bool(d["m"]),
        }
    if kind == "depth":
        return base | {
            "exch_event_ns": _ns(d.get("E"), mult),
            "exch_txn_ns": _ns(d.get("T"), mult),
            "first_update_id": int(d["U"]),
            "final_update_id": int(d["u"]),
            "prev_final_update_id": int(d["pu"]) if "pu" in d else None,
            "bids": _levels(d.get("b")),
            "asks": _levels(d.get("a")),
        }
    if kind == "mark_price":
        return base | {
            "exch_event_ns": _ns(d.get("E"), mult),
            "mark_px": _f(d.get("p")), "index_px": _f(d.get("i")),
            "settle_px": _f(d.get("P")), "funding_rate": _f(d.get("r")),
            "next_funding_ns": _ns(d.get("T"), mult),
        }
    if kind == "force_order":
        o = d.get("o", {})
        return base | {
            "exch_event_ns": _ns(d.get("E"), mult),
            "exch_trade_ns": _ns(o.get("T"), mult),
            "side": o.get("S"), "order_type": o.get("o"),
            "time_in_force": o.get("f"),
            "orig_qty": _f(o.get("q")), "px": _f(o.get("p")),
            "avg_px": _f(o.get("ap")), "status": o.get("X"),
            "last_filled_qty": _f(o.get("l")),
            "filled_accum_qty": _f(o.get("z")),
        }
    return None
 
 
def safe_lines(path: Path):
    """Yield lines, recovering across torn gzip members.
 
    These files are opened in append mode, so a restart adds a new gzip member
    to the same file. If an earlier run was killed uncleanly its member is torn
    (we want to make sure it doesn't stop here, silently skipping every GOOD member that
    was appended afterwards hiding most of an hour.)
 
    So: decompress member by member, salvage what we can from a torn one, then
    scan forward for the next valid member header and carry on.
    """
    data = path.read_bytes()
    n = len(data)
    pos, tail, torn = 0, b"", 0
 
    while pos < n:
        dec = zlib.decompressobj(31)          # 31 = gzip wrapper
        out, p, failed = [], pos, False
        while p < n and not dec.eof:
            try:
                out.append(dec.decompress(data[p:p + 262144]))
            except zlib.error:
                failed = True
                break
            p += 262144
 
        for blob in out:
            tail += blob
            *lines, tail = tail.split(b"\n")
            for ln in lines:
                yield ln.decode("utf-8", "replace")
 
        if failed:
            torn += 1
            # Skip past this member and look for the next gzip magic.
            nxt = data.find(b"\x1f\x8b\x08", pos + 1)
            if nxt == -1:
                break
            pos, tail = nxt, b""
        else:
            # End of this member = (bytes fed) - (bytes left over from the last
            # chunk). NOT n - len(unused_data): unused_data only reflects what
            # was actually fed to the decompressor, and we stop feeding as soon
            # as dec.eof goes true. Using n here overshoots to near the end of
            # the file and silently skips every later member: which is every
            # restart, since the writer appends.
            consumed = p - len(dec.unused_data)
            if consumed <= pos:               # no forward progress; bail out
                break
            pos = consumed
 
    if tail.strip():
        yield tail.decode("utf-8", "replace")
    if torn:
        log.warning("%s: %d torn gzip member(s) — recovered surrounding data",
                    path.name, torn)
 
 
def convert_file(path: Path, root: Path, force: bool) -> int:
    conn = path.parent.parent.name.split("=", 1)[1]
    stamp = path.stem.replace(".jsonl", "").rsplit("-", 1)[1]  # YYYYMMDDTHH
    date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    hour = stamp[9:11]
    mult = TS_MULT.get(conn, 1_000_000)
    venue = VENUE.get(conn, conn)
 
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    bad = 0
    for line in safe_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            sym, kind = classify(rec["st"])
            if kind == "other":
                continue
            row = normalise(kind, rec, mult)
            if row is not None:
                buckets[(sym, kind)].append(row)
        except Exception:  # noqa: BLE001 — a torn final line is expected
            bad += 1                                # after an unclean kill
    if bad:
        log.warning("%s: %d unparseable lines skipped", path.name, bad)
 
    written = 0
    for (sym, kind), rows in buckets.items():
        out_dir = (root / "curated" / f"venue={venue}" / f"symbol={sym}"
                   / f"stream={kind}" / f"date={date}" / f"hour={hour}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"part-{conn}.parquet"
        if out.exists() and not force:
            log.debug("exists, skipping %s", out)
            continue
        table = pa.Table.from_pylist(rows, schema=SCHEMAS[kind])
        pq.write_table(table, out, compression="zstd", version="2.6")
        written += len(rows)
        log.info("%s -> %d rows", out.relative_to(root), len(rows))
    return written
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--force", action="store_true", help="rewrite existing parts")
    ap.add_argument("--include-current-hour", action="store_true",
                    help="convert the hour still being written (partial data)")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
 
    root = Path(args.root).expanduser()
    current = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
 
    files = sorted((root / "raw").rglob("*.jsonl.gz"))
    total, failed = 0, []
    for f in files:
        if not args.include_current_hour and current in f.name:
            continue
        try:
            total += convert_file(f, root, args.force)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the run
            log.error("FAILED %s: %s: %s", f, type(exc).__name__, exc)
            failed.append(f)
    log.info("done: %d rows written across %d files", total, len(files))
    if failed:
        log.error("%d file(s) could not be converted:", len(failed))
        for f in failed:
            log.error("   %s", f)
 
 
if __name__ == "__main__":
    main()
