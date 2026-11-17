"""Campaign status and control.

``python -m aamd status [--convergence]``  one table of every job, the tail of
                                           WARNINGS, window overlaps and (with
                                           --convergence) the rolling half-vs-half PMF
``python -m aamd run [N]``                 start N workers (default 3) on aamd/jobs.txt
                                           unless they are already running
``python -m aamd stop``                    stop the workers (jobs resume from their
                                           last chunk when started again)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import build, checks, pmf
from .umbrella import PROTOCOL, WARNINGS, job_dir, read_jobs, workdir

JOBS = build.AAMD / "jobs.txt"
WORKERS = build.WORK / "workers.json"


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def job_state(job):
    kind, name, *rest = job
    d = job_dir(job)
    if kind == "build":
        return ("done" if (d / "system.prmtop").exists() else "queued"), {}
    if kind == "pull":
        if (d / "pull.json").exists():
            return "done", {}
        if (d / "failed_pull.json").exists():
            return "FAILED", {}
        st = d / "pull_status.json"
        lock = d / "lock"
        if lock.exists() and _alive(lock.read_text().strip() or 0) and not st.exists():
            return "running", {"job": f"pull {name}"}
        if st.exists():
            s = json.loads(st.read_text())
            stale = time.time() - s.get("updated", 0) > 1800
            return ("stalled" if stale else "running"), s
        return "queued", {}
    if (d / "failed.json").exists():
        return "FAILED", json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    if (d / "done.json").exists():
        return "done", json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    if (d / "status.json").exists():
        s = json.loads((d / "status.json").read_text())
        lock = d / "lock"
        running = lock.exists() and _alive(lock.read_text().strip() or 0)
        return ("running" if running else "paused"), s
    if (d / "lock").exists() and _alive((d / "lock").read_text().strip() or 0):
        return "equilibrating", {}
    return "queued", {}


def overlaps(name):
    """Overlap coefficient of each pair of neighbouring windows with data."""
    out = []
    series = {}
    for i, c in enumerate(PROTOCOL.centres):
        files = sorted((workdir(name) / f"w{i:02d}").glob("cv_*.npy"))
        if files:
            series[i] = np.concatenate([np.load(f)[:, 1] for f in files])
    for i in sorted(series):
        if i + 1 in series:
            out.append((float(PROTOCOL.centres[i]), float(PROTOCOL.centres[i + 1]),
                        checks.overlap(series[i], series[i + 1])))
    return out


def rolling_convergence(name):
    """(max |F_first - F_second| over 1.25-2.75 nm, the two minima) from the
    data so far, or None when some window has no data yet."""
    win = pmf.load_windows(workdir(name), PROTOCOL.centres, PROTOCOL.k_umb)
    if len(win.series) < len(PROTOCOL.centres) or min(len(s) for s in win.series) < 200:
        return None
    edges = np.arange(1.1, 2.91, 0.02)
    res = pmf.analyse(win, edges, n_blocks=2)
    m = (res.x > 1.25) & (res.x < 2.75) & np.isfinite(res.F_first) & np.isfinite(res.F_second)
    diff = float(np.max(np.abs(res.F_first[m] - res.F_second[m])))
    k1, k2 = np.nanargmin(res.F_first), np.nanargmin(res.F_second)
    ns = float(np.mean([len(s) for s in win.series]) * PROTOCOL.cv_every_ps / 1000.0)
    return {"max_diff_kT": diff, "min_first": (float(res.x[k1]), float(res.F_first[k1])),
            "min_second": (float(res.x[k2]), float(res.F_second[k2])),
            "min_all": res.minimum(), "ns_per_window": ns}


def status(convergence=False, out=sys.stdout):
    jobs = read_jobs(JOBS) if JOBS.exists() else []
    w = json.loads(WORKERS.read_text()) if WORKERS.exists() else {"pids": []}
    alive = [p for p in w.get("pids", []) if _alive(p)]
    out.write(f"aamd status  {time.strftime('%Y-%m-%d %H:%M:%S')}  workers alive: {len(alive)}/"
              f"{len(w.get('pids', []))}\n")
    counts = {}
    rows = []
    total_ns, done_ns, rates = 0.0, 0.0, []
    for job in jobs:
        st, s = job_state(job)
        counts[st] = counts.get(st, 0) + 1
        kind = job[0]
        if kind in ("window", "contact"):
            total = PROTOCOL.prod_ns if kind == "window" else float(job[3])
            total_ns += total + PROTOCOL.eq_ns
            done_ns += s.get("ns_done", 0.0) + (PROTOCOL.eq_ns if s else 0.0)
        if st in ("running", "FAILED", "paused", "stalled", "equilibrating"):
            if s.get("ns_per_day"):
                rates.append(s["ns_per_day"]) if st == "running" else None
            age = (time.time() - s["updated"]) / 60.0 if s.get("updated") else float("nan")
            rows.append(f"  {st:<13s} {' '.join(job):<32s} "
                        f"{s.get('ns_done', s.get('fraction', 0.0)):>5.2f}/{s.get('ns_total', 1.0):<4.1f} "
                        f"{(s.get('ns_per_day') or 0):6.0f} ns/d  ETA {(s.get('eta_h') or 0):5.2f} h  "
                        f"ckpt {s.get('last_checkpoint', '-')} ({age:.0f} min ago)  "
                        f"check {s.get('worst_check') or '-'}"
                        + (f"  FAILED {s.get('failed')}" if st == "FAILED" else ""))
    out.write("jobs: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n")
    agg = sum(rates)
    left = total_ns - done_ns
    out.write(f"simulated {done_ns:.1f} / {total_ns:.1f} ns; aggregate {agg:.0f} ns/day"
              + (f"; campaign ETA {left / agg:.2f} days\n" if agg else "\n"))
    out.write("\n".join(rows) + ("\n" if rows else ""))
    systems = sorted({j[1] for j in jobs if j[0] == "window"})
    for name in systems:
        ov = overlaps(name)
        low = [f"{a:.2f}-{b:.2f}: {o:.3f}" for a, b, o in ov if o < checks.OVERLAP_WARN]
        line = f"overlap {name}: {len(ov)} neighbour pairs, min {min((o for *_, o in ov), default=float('nan')):.3f}"
        out.write(line + (f"  LOW (<{checks.OVERLAP_WARN}): {low}" if low else "") + "\n")
        if convergence:
            rc = rolling_convergence(name)
            if rc is None:
                out.write(f"convergence {name}: not every window has data yet\n")
            else:
                flag = "WARN" if rc["max_diff_kT"] > checks.CONVERGENCE_WARN_KT else "ok"
                x, f, df = rc["min_all"]
                out.write(f"convergence {name}: {rc['ns_per_window']:.2f} ns/window; minimum {f:+.2f} +- {df:.2f} kT "
                          f"at {x:.2f} nm; halves {rc['min_first'][1]:+.2f} / {rc['min_second'][1]:+.2f} kT; "
                          f"max |first-second| {rc['max_diff_kT']:.2f} kT [{flag}, warn > "
                          f"{checks.CONVERGENCE_WARN_KT} kT]\n")
    if WARNINGS.exists():
        lines = WARNINGS.read_text().strip().splitlines()
        n_fail = sum(" FAIL " in l for l in lines)
        out.write(f"WARNINGS: {len(lines)} lines ({n_fail} FAIL); last 5:\n")
        for l in lines[-5:]:
            out.write("  " + l + "\n")
    else:
        out.write("WARNINGS: none\n")


def run(n=3):
    w = json.loads(WORKERS.read_text()) if WORKERS.exists() else {"pids": []}
    if any(_alive(p) for p in w.get("pids", [])):
        print("workers already running:", [p for p in w["pids"] if _alive(p)])
        return
    pids = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENMM_CPU_THREADS="1")
    for k in range(n):
        log = open(build.WORK / f"worker_{k}.log", "a")
        p = subprocess.Popen([sys.executable, "-m", "aamd.umbrella", "worker", str(k), str(n), str(JOBS)],
                             cwd=build.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        pids.append(p.pid)
    WORKERS.write_text(json.dumps({"pids": pids, "n": n, "started": time.time()}))
    print("started workers", pids)


def stop():
    w = json.loads(WORKERS.read_text()) if WORKERS.exists() else {"pids": []}
    for p in w.get("pids", []):
        if _alive(p):
            os.kill(int(p), signal.SIGTERM)
    print("sent SIGTERM to", w.get("pids", []))


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    cmd = a[0] if a else "status"
    if cmd == "status":
        status(convergence="--convergence" in a)
    elif cmd == "run":
        run(int(a[1]) if len(a) > 1 else 3)
    elif cmd == "stop":
        stop()
    else:
        print(__doc__)
