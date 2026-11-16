"""Latency benchmark for a stdio MCP memory server (Tesseract or Enigma).

Drives the real server over stdio, so every number includes JSON-RPC,
argument validation and serialisation -- what an agent actually waits on.
Counts errors and empty results per operation (a fast rejection is not a
fast answer). POSIX only (it selects on the server's pipe).

    python scripts/bench_workload.py VAULT /tmp/workload.json
    python scripts/bench_mcp.py --name tesseract --workload /tmp/workload.json \
        --env TESSERACT_VAULT_PATH=VAULT --env TESSERACT_STATE_DIR=STATE \
        -- .venv/bin/tesseract-mcp

Benchmark a copy of the vault with a copy of its state folder: a search
embeds notes that changed and saves the vectors. Compare two servers only on
the same workload file, one server at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import statistics
import subprocess
import sys
import time


class Server:
    def __init__(self, cmd: list[str], env: dict[str, str]):
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env,
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._stdin = self.proc.stdin
        self._stdout_fd = self.proc.stdout.fileno()
        self._buf = b""
        self._id = 0

    def _readline(self, timeout: float) -> bytes:
        deadline = time.perf_counter() + timeout
        fd = self._stdout_fd
        while b"\n" not in self._buf:
            left = deadline - time.perf_counter()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise TimeoutError("no response")
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise EOFError(f"server exited ({self.proc.poll()})")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def send(self, obj: dict) -> None:
        self._stdin.write(json.dumps(obj).encode() + b"\n")
        self._stdin.flush()

    def request(self, method: str, params: dict, timeout: float = 600.0):
        """Returns (response dict, seconds from send to matching response)."""
        self._id += 1
        rid = self._id
        start = time.perf_counter()
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            resp = json.loads(self._readline(timeout))
            if isinstance(resp, dict) and resp.get("id") == rid:
                return resp, time.perf_counter() - start

    def initialize(self) -> float:
        """Seconds from process spawn to the initialize response."""
        self.request("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "bench", "version": "0"},
        })
        ready = time.perf_counter() - self.t0
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return ready

    def call(self, tool: str, args: dict):
        """Returns (seconds, ok, payload_bytes, empty)."""
        resp, dt = self.request("tools/call", {"name": tool, "arguments": args})
        result = resp.get("result") or {}
        ok = "result" in resp and not result.get("isError")
        text = "".join(
            c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)
        )
        empty = text.strip() in ("", "[]", "{}", "null")
        return dt, ok, len(text), empty

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=10)


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--startups", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1, help="passes over each op's arg list")
    ap.add_argument("--out")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    workload = json.load(open(a.workload))
    first_query = workload["ops"]["search_brain"][0]

    report: dict = {"name": a.name, "cmd": cmd, "ops": {}}

    # One untimed session first: lets the server embed stale notes and fill
    # its on-disk caches, so the timed runs measure steady state, not setup.
    s = Server(cmd, env)
    s.initialize()
    dt, ok, _, _ = s.call("search_brain", first_query)
    report["priming_first_search_s"] = round(dt, 3)
    report["priming_ok"] = ok
    s.close()

    ready, first = [], []
    for _ in range(a.startups):
        s = Server(cmd, env)
        ready.append(s.initialize())
        _, ok, _, _ = s.call("search_brain", first_query)
        first.append(time.perf_counter() - s.t0)
        s.close()
        if not ok:
            print("first search failed", file=sys.stderr)
    report["startup_ready_s"] = {"median": statistics.median(ready), "max": max(ready), "n": len(ready)}
    report["startup_first_search_s"] = {"median": statistics.median(first), "max": max(first), "n": len(first)}

    s = Server(cmd, env)
    s.initialize()
    s.call("search_brain", first_query)  # warm in-process state
    for tool, arglist in workload["ops"].items():
        times, errors, empties, size = [], 0, 0, 0
        for _ in range(a.repeat):
            for args in arglist:
                dt, ok, nbytes, empty = s.call(tool, args)
                times.append(dt)
                errors += not ok
                empties += empty
                size += nbytes
        report["ops"][tool] = {
            "n": len(times),
            "median_ms": statistics.median(times) * 1000,
            "p95_ms": pct(times, 0.95) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "errors": errors,
            "empty": empties,
            "avg_bytes": size // max(1, len(times)),
        }
    s.close()

    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    print(f"== {a.name}")
    print(f"priming first search {report['priming_first_search_s']} s (ok={report['priming_ok']})")
    r, f = report["startup_ready_s"], report["startup_first_search_s"]
    print(f"startup->ready        median {r['median']*1000:8.1f} ms  max {r['max']*1000:8.1f} ms  (n={r['n']})")
    print(f"startup->first search median {f['median']*1000:8.1f} ms  max {f['max']*1000:8.1f} ms  (n={f['n']})")
    print(f"{'op':<16}{'n':>5}{'median':>10}{'p95':>10}{'min':>10}{'max':>10}{'err':>5}{'empty':>6}{'bytes':>8}")
    for tool, o in report["ops"].items():
        print(f"{tool:<16}{o['n']:>5}{o['median_ms']:>10.1f}{o['p95_ms']:>10.1f}{o['min_ms']:>10.1f}"
              f"{o['max_ms']:>10.1f}{o['errors']:>5}{o['empty']:>6}{o['avg_bytes']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
