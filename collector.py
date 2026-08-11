#!/usr/bin/env python3
"""
Binance market data collector.

Only gets bytes onto disk without losing any.

Output layout:
    <root>/raw/conn=<name>/date=YYYY-MM-DD/<name>-YYYYMMDDTHH.jsonl.gz

Each line is one received websocket message plus metadata:
    {"c": conn, "sess": session_id, "seq": n, "recv_ns": ..., "mono_ns": ...,
     "st": stream_name, "d": <original exchange payload>}
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import random
import signal
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import websockets

# --------------------------------------------------------------------------
# Endpoints.
#
# IMPORTANT: Binance split the futures websocket into routed entry points and
# decommissioned the legacy /ws and /stream paths on 2026-04-23. An unrouted
# connection still *connects* and still *accepts* your subscription, it just
# never delivers anything for streams that live under /market. 
#
#   /public  ->  bookTicker, depth, depth<levels>
#   /market  ->  aggTrade, markPrice, forceOrder, kline, ticker
#
# Spot has no such split, but spot @bookTicker carries NO exchange timestamp
# (payload is only u/s/b/B/a/A). Add timeUnit=MICROSECOND so the spot
# streams that DO carry timestamps give us microsecond resolution.
# --------------------------------------------------------------------------
SPOT_BASES = {
    "primary": "wss://stream.binance.com:9443",
    "mirror": "wss://data-stream.binance.vision",
}
FUT_BASE = "wss://fstream.binance.com"

# If a connection goes this long without ANY message, treat it as dead and
# reconnect. BTCUSDT bookTicker and markPrice@1s both tick constantly, so
# silence here always means something is wrong.
STALE_TIMEOUT_S = 30.0

FLUSH_INTERVAL_S = 5.0
HEARTBEAT_INTERVAL_S = 60.0

log = logging.getLogger("collector")


@dataclass
class Conn:
    name: str
    url: str
    expect: list[str]  # stream names we assert are actually arriving


def build_connections(spot_base: str, symbols: list[str], depth: bool) -> list[Conn]:
    syms = [s.lower() for s in symbols]

    spot_streams: list[str] = []
    for s in syms:
        spot_streams += [f"{s}@bookTicker", f"{s}@aggTrade"]
        if depth:
            spot_streams.append(f"{s}@depth@100ms")

    fut_public: list[str] = []
    for s in syms:
        fut_public.append(f"{s}@bookTicker")
        if depth:
            fut_public.append(f"{s}@depth@100ms")

    fut_market: list[str] = []
    for s in syms:
        fut_market += [f"{s}@aggTrade", f"{s}@markPrice@1s", f"{s}@forceOrder"]

    return [
        Conn(
            "spot",
            f"{spot_base}/stream?streams={'/'.join(spot_streams)}&timeUnit=MICROSECOND",
            # forceOrder is sparse; every stream listed here must be frequent
            # enough that silence is genuinely a fault.
            expect=[x for x in spot_streams],
        ),
        Conn(
            "fut_public",
            f"{FUT_BASE}/public/stream?streams={'/'.join(fut_public)}",
            expect=list(fut_public),
        ),
        Conn(
            "fut_market",
            f"{FUT_BASE}/market/stream?streams={'/'.join(fut_market)}",
            # forceOrder can legitimately be silent for hours — excluded from
            # the "must be arriving" assertion, but still counted.
            expect=[x for x in fut_market if not x.endswith("@forceOrder")],
        ),
    ]


class HourlyWriter:
    """Append-only gzip writer that rotates on the UTC hour."""

    def __init__(self, root: Path, conn_name: str):
        self.root = root
        self.conn_name = conn_name
        self._hour_key: str | None = None
        self._fh = None
        self._dirty = False

    def _path_for(self, dt: datetime) -> Path:
        d = dt.strftime("%Y-%m-%d")
        stamp = dt.strftime("%Y%m%dT%H")
        p = self.root / "raw" / f"conn={self.conn_name}" / f"date={d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{self.conn_name}-{stamp}.jsonl.gz"

    def _rotate_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        key = now.strftime("%Y%m%dT%H")
        if key != self._hour_key:
            if self._fh is not None:
                self._fh.close()
            path = self._path_for(now)
            # Append mode: if the process restarts mid-hour then add to the
            # existing file rather than truncating it. gzip handles
            # concatenated members transparently on read.
            self._fh = gzip.open(path, "at", compresslevel=4, encoding="utf-8")
            self._hour_key = key
            log.info("[%s] writing to %s", self.conn_name, path)

    def write(self, line: str) -> None:
        self._rotate_if_needed()
        self._fh.write(line)
        self._fh.write("\n")
        self._dirty = True

    def flush(self) -> None:
        if self._fh is not None and self._dirty:
            self._fh.flush()
            self._dirty = False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Stats:
    """Counters keyed by (connection, stream).

    The connection MUST be part of the key: btcusdt@bookTicker exists on both
    spot and futures, so keying by stream name alone merges the two venues and
    makes the heartbeat report phantom dead streams.
    """

    def __init__(self):
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.last_ns: dict[tuple[str, str], int] = {}
        self.reconnects: dict[str, int] = defaultdict(int)
        self.gaps: list[tuple[str, float]] = []

    def record(self, conn: str, stream: str, recv_ns: int) -> None:
        self.counts[(conn, stream)] += 1
        self.last_ns[(conn, stream)] = recv_ns


async def run_connection(conn: Conn, writer: HourlyWriter, stats: Stats,
                         stop: asyncio.Event) -> None:
    backoff = 1.0
    disconnected_at: float | None = None

    while not stop.is_set():
        session = uuid.uuid4().hex[:12]
        seq = 0
        try:
            # ping_interval=None: No client pings. Binance sends
            # its own ping frames and the library auto-replies with a pong,
            # which is what keeps the connection alive. Liveness is
            # enforced by STALE_TIMEOUT_S on recv().
            async with websockets.connect(
                conn.url,
                ping_interval=None,
                ping_timeout=None,
                max_queue=None,
                open_timeout=15,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
            ) as ws:
                if disconnected_at is not None:
                    gap = time.monotonic() - disconnected_at
                    stats.gaps.append((conn.name, gap))
                    log.warning("[%s] reconnected after %.2fs gap", conn.name, gap)
                    disconnected_at = None
                log.info("[%s] connected session=%s", conn.name, session)
                backoff = 1.0

                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=STALE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        log.error("[%s] no data for %.0fs — forcing reconnect",
                                  conn.name, STALE_TIMEOUT_S)
                        break

                    recv_ns = time.time_ns()
                    mono_ns = time.monotonic_ns()

                    try:
                        env = json.loads(msg)
                    except json.JSONDecodeError:
                        log.error("[%s] undecodable frame, skipped", conn.name)
                        continue

                    # Combined streams wrap payloads as {"stream":..,"data":..}.
                    # Subscription acks arrive as {"result":null,"id":..}.
                    if "stream" not in env:
                        log.info("[%s] control frame: %s", conn.name, msg[:200])
                        continue

                    stream = env["stream"]
                    rec = {
                        "c": conn.name,
                        "sess": session,
                        "seq": seq,
                        "recv_ns": recv_ns,
                        "mono_ns": mono_ns,
                        "st": stream,
                        "d": env["data"],
                    }
                    writer.write(json.dumps(rec, separators=(",", ":")))
                    stats.record(conn.name, stream, recv_ns)
                    seq += 1

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — never let one conn kill the run
            log.error("[%s] %s: %s", conn.name, type(exc).__name__, exc)

        if stop.is_set():
            break
        if disconnected_at is None:
            disconnected_at = time.monotonic()
        stats.reconnects[conn.name] += 1
        sleep = min(backoff, 30.0) * (0.5 + random.random())
        log.info("[%s] retrying in %.1fs", conn.name, sleep)
        await asyncio.sleep(sleep)
        backoff = min(backoff * 2, 30.0)


async def flusher(writers: list[HourlyWriter], stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(FLUSH_INTERVAL_S)
        for w in writers:
            try:
                w.flush()
            except Exception as exc:  # noqa: BLE001
                log.error("flush failed for %s: %s", w.conn_name, exc)


async def heartbeat(conns: list[Conn], stats: Stats, stop: asyncio.Event) -> None:
    """Assert on per-stream message counts, and keep asserting."""
    prev: dict[tuple[str, str], int] = defaultdict(int)
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        now_ns = time.time_ns()
        parts, dead = [], []
        for conn in conns:
            for stream in conn.expect:
                key = (conn.name, stream)
                total = stats.counts.get(key, 0)
                delta = total - prev[key]
                prev[key] = total
                parts.append(f"{conn.name}/{stream}={delta}")
                last = stats.last_ns.get(key)
                age = (now_ns - last) / 1e9 if last else float("inf")
                if delta == 0 or age > STALE_TIMEOUT_S:
                    dead.append(f"{conn.name}/{stream}(age={age:.0f}s)")
        log.info("1m counts: %s", " ".join(parts))
        if dead:
            log.error("SILENT STREAMS: %s — check endpoint routing", ", ".join(dead))


async def main_async(args) -> None:
    root = Path(args.root).expanduser()
    conns = build_connections(SPOT_BASES[args.spot_base], args.symbols, args.depth)

    for c in conns:
        log.info("[%s] %s", c.name, c.url)

    writers = [HourlyWriter(root, c.name) for c in conns]
    stats = Stats()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(run_connection(c, w, stats, stop))
        for c, w in zip(conns, writers)
    ]
    tasks.append(asyncio.create_task(flusher(writers, stop)))
    tasks.append(asyncio.create_task(heartbeat(conns, stats, stop)))

    await stop.wait()
    log.info("shutting down")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for w in writers:
        w.flush()
        w.close()
    for (cname, stream), n in sorted(stats.counts.items()):
        log.info("total %s/%s = %d", cname, stream, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data", help="output directory")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--spot-base", choices=list(SPOT_BASES), default="primary")
    ap.add_argument("--depth", action="store_true",
                    help="also capture @depth@100ms diffs")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)sZ %(levelname)s %(message)s",
    )
    logging.Formatter.converter = time.gmtime

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
