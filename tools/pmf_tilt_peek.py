"""Angular falloff and alignment torque from the tilt series."""
import collections
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "pmf_data", "forces_tilt.csv"))))

# pair contact (gap 1.5) with its baseline (gap 6.0) per context
ctx = {}
for r in rows:
    key = (r["sA"], r["delta"], r["alpha"], r["beta"], r["tilt"])
    ctx.setdefault(key, {})[float(r["gap"])] = r

# per-context series: normalize each context's force to its own tilt=0
series = collections.defaultdict(dict)   # (sA,delta,a,b) -> tilt -> (Fx, Tx)
for key, g in ctx.items():
    if 1.5 in g and 6.0 in g:
        c = (key[0], key[1], key[2], key[3])
        series[c][float(key[4])] = (
            float(g[1.5]["Fx"]) - float(g[6.0]["Fx"]),
            float(g[1.5]["Tx"]) - float(g[6.0]["Tx"]))

ratios = collections.defaultdict(list)
torques = collections.defaultdict(list)
nctx = 0
for c, sv in series.items():
    if 0.0 not in sv or abs(sv[0.0][0]) < 5.0:
        continue     # need a solid tilt-0 reference to normalize against
    nctx += 1
    f0 = sv[0.0][0]
    for t, (f, tx) in sv.items():
        ratios[t].append(f / f0)
        # torque signed toward restoring: positive = pushes back to parallel
        torques[t].append(-math.copysign(tx, t) if t > 0 else tx)

def stats(v):
    m = sum(v) / len(v)
    sem = (math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
           / math.sqrt(max(1, len(v))))
    return m, sem

print(f"{nctx} contexts with |F(0)| > 5 kT/nm")
print(f"{'tilt(deg)':>9} {'n':>3} {'F/F0':>7} {'sem':>5} {'cos^2':>6} "
      f"{'restoringTx':>11} {'sem':>5}")
for t in sorted(ratios):
    m, s = stats(ratios[t])
    mt, st_ = stats(torques[t])
    print(f"{math.degrees(t):9.0f} {len(ratios[t]):3d} {m:7.2f} {s:5.2f} "
          f"{math.cos(t)**2:6.2f} {mt:11.2f} {st_:5.2f}")
