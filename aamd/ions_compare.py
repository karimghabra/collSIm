"""Ion bridges: all-atom counts against the modcoll ion sampler at the same
geometry.

``python -m aamd.ions_compare SYSTEM RUN [--frames N] [--sweeps N]``  (RUN: w07, c1.300 or a path)

1. All-atom: every saved frame of the job (solute + ions, 10 ps apart) is
   analysed with ``aamd.bridges`` (modcoll's contact and bridge definitions);
   bridges per frame by kind, ions in contact with protein, and the bulk
   concentration of the divalent ion (ions with no solute heavy atom within
   1 nm, over the volume farther than 1 nm from the solute).
2. Coarse: for N frames spread over the job, the protein groups at their
   all-atom positions become a modcoll LigandSet (``bridges.ligandset_from_frame``),
   the two helix cores z-parallel excluded cylinders of radius 0.5 nm (the
   value of modcoll's S3 rows) at the segments' axes, and modcoll's grand-
   canonical sampler is run in the same box at the all-atom bulk concentration
   (modcoll's bath convention: free-ion concentration) and the monovalent
   ionic strength of the box. Bridges are counted with modcoll's own
   ``BridgeAnalyzer`` every 10 sweeps.

Output: ``aamd/results/ions_SYSTEM_JOB.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from . import bridges as B
from . import build

RESULTS = build.AAMD / "results"
CORE_RADIUS_NM = 0.5
SOLUTE_EXCLUDE = ("Na+", "Cl-", "MG", "CA", "SO4", "WAT", "HOH")


def segment_mapper(topology, n_res=30):
    """residue -> 0 (segment A), 1 (segment B) or None, by solute residue order."""
    prot = [r for r in topology.residues() if r.name not in SOLUTE_EXCLUDE]
    per = 3 * (n_res + 2)
    seg = {r.index: (0 if k < per else 1) for k, r in enumerate(prot)}
    return lambda res: seg.get(res.index)


def load_job(jobdir: Path):
    import mdtraj

    jobdir = Path(jobdir)
    dcds = sorted(glob.glob(str(jobdir / "solute_*.dcd")))
    traj = mdtraj.load(dcds, top=str(jobdir / "solute.pdb"))
    return traj


def bulk_concentration(xyz, box, ion_idx, solute_heavy, rng, cutoff=1.0, n_probe=4000):
    """Divalent-ion concentration (mol/L) away from the solute in one frame."""
    if len(ion_idx) == 0:
        return np.nan
    from scipy.spatial import cKDTree

    sol = np.mod(xyz[solute_heavy], box)
    tree = cKDTree(sol, boxsize=box)
    d_ion, _ = tree.query(np.mod(xyz[ion_idx], box))
    probes = rng.random((n_probe, 3)) * box
    d_p, _ = tree.query(probes)
    vol = float(np.prod(box)) * float(np.mean(d_p > cutoff))          # nm^3
    n_bulk = int(np.count_nonzero(d_ion > cutoff))
    return n_bulk / (vol * 0.602214076)


def md_counts(traj, rng, thr=None):
    thr = B.Thresholds.from_params() if thr is None else thr
    top = traj.topology.to_openmm()
    seg = segment_mapper(top)
    part = B.find_partners(top, seg)
    ions = B.find_ions(top)
    species = sorted({i[0] for i in ions})
    heavy = np.array([a.index for a in top.atoms() if a.residue.name not in SOLUTE_EXCLUDE
                      and a.element is not None and a.element.symbol != "H"])
    div_idx = [i[1] for i in ions]
    per_frame, bulk, bound = [], [], []
    for f in range(traj.n_frames):
        xyz = traj.xyz[f].astype(float)
        box = traj.unitcell_lengths[f].astype(float)
        br = B.bridges(xyz, box, ions, part, thr)
        counts = {}
        for b in br:
            counts[B.kind(b)] = counts.get(B.kind(b), 0) + 1
        per_frame.append(counts)
        c = B.contacts(xyz, box, ions, part, thr)
        bound.append(len({row[0] for row in c}))
        if f % 5 == 0:
            bulk.append(bulk_concentration(xyz, box, div_idx, heavy, rng))
    kinds = ["inner-inner", "inner-outer", "outer-outer"]
    arr = np.array([[pf.get(k, 0) for k in kinds] for pf in per_frame], float)
    return {"species": species, "n_frames": traj.n_frames, "kinds": kinds,
            "bridges_mean": arr.mean(axis=0).tolist(), "bridges_total_mean": float(arr.sum(axis=1).mean()),
            "bridges_total_block_se": block_se(arr.sum(axis=1)),
            "ions_in_contact_mean": float(np.mean(bound)), "n_divalent": len(div_idx),
            "bulk_conc_M": float(np.nanmean(bulk)) if bulk else float("nan"),
            "partners": {"carboxylate_O": int(np.sum(part.kind == "O")),
                         "donor_N": int(np.sum(part.kind != "O"))}}


def block_se(x, n_blocks=5):
    x = np.asarray(x, float)
    if len(x) < n_blocks:
        return float("nan")
    m = [b.mean() for b in np.array_split(x, n_blocks)]
    return float(np.std(m, ddof=1) / np.sqrt(n_blocks))


def coarse_counts(traj, frames, species, conc_M, ionic_M, sweeps=6000, equil=2000, seed=20261117):
    from modcoll.ions import BridgeAnalyzer, IonSampler, load_params

    params = load_params()
    top = traj.topology.to_openmm()
    seg = segment_mapper(top)
    ca = {s: [a.index for a in top.atoms() if a.name == "CA" and seg(a.residue) == s] for s in (0, 1)}
    kinds = ["inner-inner", "inner-outer", "outer-outer"]
    per_frame = []
    for n, f in enumerate(frames):
        xyz = traj.xyz[f].astype(float)
        box = traj.unitcell_lengths[f].astype(float)
        lig = B.ligandset_from_frame(xyz, top, seg)
        lig = type(lig)(np.mod(lig.pos, box), lig.type, lig.charge, lig.mol, lig.group, lig.normal)
        cyl = np.array([[*np.mod(xyz[ca[s]].mean(axis=0)[:2], box[:2]), CORE_RADIUS_NM] for s in (0, 1)])
        sampler = IonSampler(params, lig, box, [True] * 3, ionic_M, np.random.default_rng([seed, n]),
                             bath={species: conc_M}, cylinders=cyl, local_moves_per_ion=4)
        an = BridgeAnalyzer(params, lig, box, [True] * 3)
        sampler.sweep(equil)
        counts = []
        for _ in range(sweeps // 10):
            sampler.sweep(10)
            c = np.zeros(3)
            for b in an.bridges(sampler.snapshot()):
                c[kinds.index(b.kind)] += 1
            counts.append(c)
        per_frame.append(np.mean(counts, axis=0))
    arr = np.array(per_frame)
    return {"frames": [int(f) for f in frames], "kinds": kinds, "bridges_mean": arr.mean(axis=0).tolist(),
            "bridges_total_mean": float(arr.sum(axis=1).mean()),
            "bridges_total_frame_sd": float(arr.sum(axis=1).std(ddof=1)) if len(arr) > 1 else float("nan"),
            "sweeps": sweeps, "equil_sweeps": equil, "bath_M": conc_M, "ionic_strength_M": ionic_M,
            "core_radius_nm": CORE_RADIUS_NM}


def compare(name, jobdir, n_frames=8, sweeps=6000):
    from .predict import ionic_strength

    rng = np.random.default_rng(1)
    if not Path(jobdir).exists():                  # a run name such as w07 or c1.300
        jobdir = build.WORK / name / jobdir
    traj = load_job(jobdir)
    md = md_counts(traj, rng)
    sp = [s for s in md["species"]]
    if len(sp) != 1:
        raise ValueError(f"expected one divalent species, found {sp}")
    conc = md["bulk_conc_M"] if np.isfinite(md["bulk_conc_M"]) and md["bulk_conc_M"] > 0 else 0.05
    frames = np.linspace(0, traj.n_frames - 1, n_frames).astype(int)
    cg = coarse_counts(traj, frames, sp[0], conc, ionic_strength(name), sweeps=sweeps)
    out = {"system": name, "job": Path(jobdir).name, "md": md, "coarse": cg}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ions_{name}_{Path(jobdir).name}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("jobdir")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=6000)
    a = ap.parse_args(argv)
    o = compare(a.system, a.jobdir, a.frames, a.sweeps)
    print(json.dumps({"md": {k: o["md"][k] for k in ("bridges_mean", "bridges_total_mean", "bulk_conc_M",
                                                       "ions_in_contact_mean")},
                      "coarse": {k: o["coarse"][k] for k in ("bridges_mean", "bridges_total_mean")}}, indent=1))


if __name__ == "__main__":
    main()
