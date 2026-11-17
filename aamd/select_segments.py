"""How segment A (helix residues 624-653) was chosen, with the coarse model.

``python -m aamd.select_segments`` writes ``aamd/results/segment_scan.csv``.

For windows of 30 residues starting every 6 residues (from residue 240, so
that the partner at 1 D exists), and for staggers 1, 0.5, 0.75 and 0.25 D,
the coarse free-rotation energy (``aamd.coarse``, 12 x 12 rotations) of the
native pair at pH 7.4 and of the fully succinylated pair at pH 4.2, both with
Henderson-Hasselbalch mean charges, at d = 1.65 nm and 150 mM. The window
kept is the one with the largest margin between 1 D and the most favourable
off-D stagger for the native pair.
"""
from __future__ import annotations

import csv

import numpy as np

from modcoll.sites import native_molecule

from . import build, coarse

STAGGERS = {"1D": 234, "0.5D": 117, "0.75D": 175, "0.25D": 58}


def scan(n_res=30, step=6, d=1.65, n_angle=12):
    rise = native_molecule().rise_nm
    rows = []
    for i in range(240, 1014 - n_res + 1, step):
        a = coarse.segment(i, n_res)
        row = {"start": i, "n_lys": int(np.sum(a.site_type == 4))}
        for tag, st in STAGGERS.items():
            b = coarse.segment(i - st, n_res)
            for chem, (sa, sb, pH) in {"native": (a, b, 7.4),
                                       "succ": (coarse.succinylate(a), coarse.succinylate(b), 4.2)}.items():
                el, hy = coarse.energy_grid(sa, coarse.mean_charges(sa, pH), sb, coarse.mean_charges(sb, pH),
                                            st * rise, [d], n_angle)
                row[f"{chem}_{tag}"] = float(coarse.reduce_boltzmann(el + hy)[0])
        row["native_margin"] = min(row[f"native_{t}"] for t in STAGGERS if t != "1D") - row["native_1D"]
        rows.append(row)
    return rows


def main():
    rows = scan()
    build.AAMD.joinpath("results").mkdir(exist_ok=True)
    with open(build.AAMD / "results" / "segment_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["native_margin"])
    print(f"best window: residues {best['start']}-{best['start'] + 29}, margin {best['native_margin']:.2f} kT "
          f"(native 1D {best['native_1D']:+.2f}, 0.5D {best['native_0.5D']:+.2f})")


if __name__ == "__main__":
    main()
