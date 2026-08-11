"""Replicate-averaged angular falloff from the deep tilt series."""
import collections
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.path.join(ROOT, "pmf_data")

# baselines: far-gap rows from both tilt files, keyed (sA, delta, a, b, tilt)
base = {}
for fn in ("forces_tilt.csv", "forces_tiltdeep.csv"):
    path = os.path.join(PD, fn)
    if not os.path.exists(path):
        continue
    for r in csv.DictReader(open(path)):
        if float(r["gap"]) > 5.0:
            base[(float(r["sA"]), float(r["delta"]), float(r["alpha"]),
                  float(r["beta"]), float(r["tilt"]))] = (float(r["Fx"]),
                                                          float(r["Tx"]))

# contact replicates from the deep file
reps = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(PD, "forces_tiltdeep.csv"))):
    if float(r["gap"]) > 5.0:
        continue
    key = (float(r["sA"]), float(r["delta"]), float(r["alpha"]),
           float(r["beta"]), float(r["tilt"]))
    reps[key].append((float(r["Fx"]), float(r["Tx"])))

# per-context normalized series
byctx = collections.defaultdict(dict)
for (sA, dz, a, b, t), v in reps.items():
    bl = base.get((sA, dz, a, b, t))
    if bl is None:
        continue
    fx = sum(x[0] for x in v) / len(v) - bl[0]
    tx = sum(x[1] for x in v) / len(v) - bl[1]
    sem = 0.0
    if len(v) > 1:
        m = sum(x[0] for x in v) / len(v)
        sem = math.sqrt(sum((x[0] - m) ** 2 for x in v) / (len(v) - 1) / len(v))
    byctx[(sA, dz, a, b)][t] = (fx, tx, sem, len(v))

print(f"{'ctx':>18} {'tilt(deg)':>9} {'Fx':>8} {'sem':>5} {'F/F0':>6} "
      f"{'cos^2':>6} {'Tx':>8}")
agg = collections.defaultdict(list)
for c, sv in sorted(byctx.items()):
    if 0.0 not in sv:
        continue
    f0 = sv[0.0][0]
    for t in sorted(sv):
        fx, tx, sem, n = sv[t]
        r = fx / f0 if abs(f0) > 1e-9 else float("nan")
        agg[t].append(r)
        print(f"{str(c[:2]):>18} {math.degrees(t):9.0f} {fx:8.2f} {sem:5.2f} "
              f"{r:6.2f} {math.cos(t)**2:6.2f} {tx:8.2f}")
print()
print(f"{'tilt(deg)':>9} {'mean F/F0':>9} {'cos^2':>6}")
for t in sorted(agg):
    v = [x for x in agg[t] if x == x]
    print(f"{math.degrees(t):9.0f} {sum(v)/len(v):9.2f} {math.cos(t)**2:6.2f}")
