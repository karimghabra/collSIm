"""The coarse model's prediction for every all-atom system.

``python -m aamd.predict [SYSTEM ...]`` writes ``aamd/results/predict_SYSTEM.json``.

For each system: the same two segments, the same discrete protonation
pattern (``build.segment_types``, same seeds) and the monovalent ionic
strength of the all-atom box (Na+ and Cl-, counterions included, per litre of
water), and the modcoll site energy as a function of inter-axial separation d:

* ``F_free``: -ln < exp(-E(d, a, b)) > over 24 x 24 rotations of the two
  segments about their own axes. Both segments rotate freely in the all-atom
  runs, so this is the coarse counterpart of the measured PMF (the coarse
  model has no excluded volume: it says nothing about the wall at contact);
* ``E_min``: the most favourable rotation pair;
* the same split into electrostatic and hydrophobic parts (each reduced on
  its own, so the parts need not add up to the total);
* ``F_free_hh``: the same with modcoll's Henderson-Hasselbalch mean charges
  instead of the discrete pattern.

The divalent ions are not part of the site model (they are explicit particles
in modcoll.ions); for those systems only the monovalent background enters here
and the ion comparison is done on bridge counts (``aamd.ions_compare``).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from modcoll.sites import native_molecule

from . import build, coarse

D_GRID = np.round(np.arange(1.20, 3.001, 0.05), 3)
RESULTS = build.AAMD / "results"


def ionic_strength(name) -> float:
    rec = json.loads((build.WORK / name / "build.json").read_text())
    per_M = rec["n_water_pass1"] / build.WATER_MOLARITY
    ions = rec["ions"]
    return 0.5 * (ions.get("Na+", 0) + ions.get("Cl-", 0)) / per_M


def predict(name, n_angle=24):
    spec = build.SYSTEMS[name]
    seg_a, qa = build.segment_types(spec, "A")
    seg_b, qb = build.segment_types(spec, "B")
    stagger = spec.stagger_res * native_molecule().rise_nm
    I = ionic_strength(name) if (build.WORK / name / "build.json").exists() else spec.salt_M
    params = coarse.CoarseParams(ionic_strength_M=I)
    el, hy = coarse.energy_grid(seg_a, qa, seg_b, qb, stagger, D_GRID, n_angle, params)
    qa_hh, qb_hh = coarse.mean_charges(seg_a, spec.pH), coarse.mean_charges(seg_b, spec.pH)
    el_hh, _ = coarse.energy_grid(seg_a, qa_hh, seg_b, qb_hh, stagger, D_GRID, n_angle, params)
    tot = el + hy
    out = {
        "system": name, "d_nm": D_GRID.tolist(), "ionic_strength_M": I,
        "debye_nm": params.debye_nm, "eps_h": params.eps_h,
        "net_charge": [float(qa.sum()), float(qb.sum())],
        "F_free": coarse.reduce_boltzmann(tot).tolist(),
        "E_min": coarse.reduce_min(tot).tolist(),
        "F_free_el": coarse.reduce_boltzmann(el).tolist(),
        "F_free_hy": coarse.reduce_boltzmann(hy).tolist(),
        "E_mean_el": el.reshape(len(D_GRID), -1).mean(axis=1).tolist(),
        "E_mean_hy": hy.reshape(len(D_GRID), -1).mean(axis=1).tolist(),
        "F_free_hh": coarse.reduce_boltzmann(el_hh + hy).tolist(),
        "hydrophobic_overlap": (hy.reshape(len(D_GRID), -1).mean(axis=1) / -params.eps_h).tolist(),
        "n_angle": n_angle,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"predict_{name}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    names = (sys.argv[1:] if argv is None else argv) or [n for n in build.SYSTEMS
                                                         if (build.WORK / n / "build.json").exists()]
    for n in names:
        o = predict(n)
        k = int(np.argmin(o["F_free"]))
        print(f"{n:<16s} I={o['ionic_strength_M']:.3f} M  F_free min {o['F_free'][k]:+.2f} kT at {o['d_nm'][k]:.2f} nm;"
              f" at 1.65 nm F_free {o['F_free'][9]:+.2f} (el {o['F_free_el'][9]:+.2f}, hy {o['F_free_hy'][9]:+.2f})")


if __name__ == "__main__":
    main()
