#!/usr/bin/env python3
"""
Continuity and quality report over the raw capture.

Reports, per connection:
  * hourly message counts per stream (a zero row is a dead stream)
  * every gap longer than --gap-ms, with its duration and the session boundary
  * exchange-vs-local latency percentiles (where an exchange timestamp exists)
  * total wall-clock span vs. sum of covered time

Stdlib only, so it runs anywhere the collector runs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# Reuse the torn-member-tolerant reader rather than duplicating it. These files
# are append-mode gzip, so an unclean kill leaves a torn member with good data
# behind it, we don't want a reader either crashiing or silently skipping the
# remainder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_parquet import safe_lines  # noqa: E402

TS_MULT = {"spot": 1_000, "fut_public": 1_000_000, "fut_market": 1_000_000}


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def fmt_ts(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--gap-ms", type=float, default=2000.0,
                    help="report gaps in local receive time longer than this")
    ap.add_argument("--conn", help="restrict to one connection")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    files = sorted((root / "raw").rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no raw files under {root/'raw'}")

    per_conn_hour = defaultdict(lambda: defaultdict(int))   # (conn,hour) -> stream -> n
    latency = defaultdict(list)                             # conn -> [ms]
    gaps = defaultdict(list)                                # conn -> [(t, dur_ms, note)]
    sessions = defaultdict(set)
    first_ns, last_ns = {}, {}
    prev_recv, prev_sess = {}, {}

    for path in files:
        conn = path.parent.parent.name.split("=", 1)[1]
        if args.conn and conn != args.conn:
            continue
        mult = TS_MULT.get(conn, 1_000_000)
        hour = path.stem.replace(".jsonl", "").rsplit("-", 1)[1]

        for line in safe_lines(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue

            recv = r["recv_ns"]
            per_conn_hour[(conn, hour)][r["st"]] += 1
            sessions[conn].add(r["sess"])
            first_ns.setdefault(conn, recv)
            last_ns[conn] = recv

            # latency: only meaningful where the exchange sends a timestamp.
            E = r["d"].get("E")
            if E is not None:
                latency[conn].append((recv - int(E) * mult) / 1e6)

            p = prev_recv.get(conn)
            if p is not None:
                dur_ms = (recv - p) / 1e6
                if dur_ms > args.gap_ms:
                    note = ("reconnect" if prev_sess.get(conn) != r["sess"]
                            else "in-session stall")
                    gaps[conn].append((p, dur_ms, note))
            prev_recv[conn] = recv
            prev_sess[conn] = r["sess"]

    for conn in sorted(sessions):
        span_h = (last_ns[conn] - first_ns[conn]) / 1e9 / 3600
        lost_s = sum(d for _, d, _ in gaps[conn]) / 1000
        print(f"\n{'='*72}\nCONNECTION: {conn}")
        print(f"  span      : {fmt_ts(first_ns[conn])} -> {fmt_ts(last_ns[conn])} "
              f"UTC  ({span_h:.2f} h)")
        print(f"  sessions  : {len(sessions[conn])} "
              f"({len(sessions[conn]) - 1} reconnects)")
        print(f"  gaps>{args.gap_ms:.0f}ms: {len(gaps[conn])}  "
              f"total {lost_s:.1f}s  ({lost_s / max(span_h*36, 1e-9):.3f}% of span)")

        lat = sorted(latency[conn])
        if lat:
            print(f"  latency ms: p50={pct(lat,50):.1f} p90={pct(lat,90):.1f} "
                  f"p99={pct(lat,99):.1f} max={lat[-1]:.1f}")
            if pct(lat, 50) < -5:
                print("  !! negative median latency — your clock is behind the "
                      "exchange. Check chrony/NTP.")

        print("  hourly counts:")
        hours = sorted(h for (c, h) in per_conn_hour if c == conn)
        streams = sorted({s for (c, h), d in per_conn_hour.items()
                          if c == conn for s in d})
        head = "    " + "hour".ljust(14) + "".join(s.ljust(24) for s in streams)
        print(head)
        for h in hours:
            row = per_conn_hour[(conn, h)]
            cells = "".join(str(row.get(s, 0)).ljust(24) for s in streams)
            flag = "   <-- ZERO" if any(row.get(s, 0) == 0 for s in streams) else ""
            print("    " + h.ljust(14) + cells + flag)

        for t, d, note in gaps[conn][:20]:
            print(f"    GAP {fmt_ts(t)} UTC  {d/1000:.2f}s  ({note})")
        if len(gaps[conn]) > 20:
            print(f"    ... and {len(gaps[conn]) - 20} more")


if __name__ == "__main__":
    main()
