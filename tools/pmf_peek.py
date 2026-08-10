"""Quick look at the accumulating PMF data: mean axial force vs stagger."""
import collections
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "pmf_data", "forces.csv"))))
g = collections.defaultdict(list)
for r in rows:
    if float(r["ns"]) >= 0.9 and abs(float(r["gap"]) - 1.5) < 0.01:
        g[round(float(r["delta"]), 1)].append(float(r["Fz"]))

print(f"{'delta':>7} {'n':>3} {'meanFz':>8} {'sem':>6}   (Fz>0 pushes toward larger stagger)")
for d in sorted(g):
    v = g[d]
    m = sum(v) / len(v)
    sem = (math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
           / math.sqrt(len(v)))
    bar = "#" * min(30, int(abs(m)))
    print(f"{d:7.1f} {len(v):3d} {m:8.2f} {sem:6.2f}  {'+' if m > 0 else '-'}{bar}")
