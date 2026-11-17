"""Sanity checks run on every trajectory chunk as it is written.

Each check returns a ``Check`` with the measured value, the thresholds and a
verdict: "ok", "warn" (one line goes to the campaign's WARNINGS file) or
"fail" (a hard failure: the window stops, is marked failed, and the worker
moves on to its next job). The thresholds are stated here and nowhere else;
docs/explicit_solvent_pmf.md copies this table.

=======================  ====================================================  ===============  ===============
check                    what is measured (per 0.5 ns chunk)                    warn             fail
=======================  ====================================================  ===============  ===============
finite                   NaN or inf in energies, CV or the final coordinates    -                any
temperature              mean kinetic temperature minus 298.15 K (the estimator |dT| > 2 K       |dT| > 5 K
                         reads about +1 K at 4 fs: half-step kinetic energy)
temperature_peak         largest instantaneous kinetic temperature              -                > 400 K
energy_drift             change of the mean potential energy since the          > 0.3 %          > 2 %
                         previous chunk, relative to its magnitude
volume_drift             change of the mean box volume since the previous       > 1 %            > 3 %
                         chunk
bond_stretch             longest bond between solute heavy atoms, any frame     > 0.20 nm        > 0.25 nm
helix_rmsd_local         backbone (N, CA, C) RMSD to the built helix of every   > 0.20 nm        > 0.30 nm
                         block of two triplets (6 residues, three chains),
                         each block superposed on its own; the largest, chunk
                         mean (fraying or local unfolding)
helix_rmsd_global        the same over the whole segment (includes bending and  > 0.50 nm        -
                         twist relaxation of the 9 nm segment away from the
                         ideal (Pro-Pro-Gly)10 template)
interchain_hbonds        backbone N-H...O=C hydrogen bonds between chains of    < 60 % of built  < 40 % of built
                         one segment (N...O < 0.35 nm, N-H...O > 120 deg),
                         chunk mean, smallest of the two segments
umbrella_centre          mean separation minus the window centre                |d| > 0.15 nm    |d| > 0.30 nm
lateral_offset           mean |Delta y| and mean |Delta z - z0| of the centroids > 0.05 nm         > 0.15 nm
ion_count                ions of each kind present vs. the build record          -                any difference
ion_coordination         mean number of O within 0.26 nm (Mg) / 0.30 nm (Ca)    Mg outside       Mg outside
                         of each divalent cation, final frame                   5.5-6.5, Ca      4-8, Ca
                                                                                outside 6-8.5    outside 4-10
=======================  ====================================================  ===============  ===============

Campaign-level checks (``aamd status``): neighbouring-window histogram overlap
(sum of min(p, q) over 0.01 nm bins; warn below 0.03) and rolling convergence
(the PMF from the first and from the second half of the data so far; warn when
they differ by more than 1 kT anywhere in 1.25-2.75 nm).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

T_TARGET = 298.15


@dataclasses.dataclass
class Check:
    name: str
    value: float
    warn: str
    fail: str
    verdict: str           # "ok", "warn", "fail"
    detail: str = ""

    def line(self, where: str) -> str:
        return (f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {where} {self.name} {self.verdict.upper()} "
                f"value={self.value:.4g} warn[{self.warn}] fail[{self.fail}] {self.detail}").rstrip()


def _grade(name, value, warn_if, fail_if, warn_txt, fail_txt, detail=""):
    if not np.isfinite(value):
        return Check(name, float(value), warn_txt, fail_txt, "fail", "not finite " + detail)
    verdict = "fail" if fail_if(value) else ("warn" if warn_if(value) else "ok")
    return Check(name, float(value), warn_txt, fail_txt, verdict, detail)


# ----------------------------------------------------------- scalar series ---
def check_finite(*arrays) -> Check:
    bad = sum(int(np.count_nonzero(~np.isfinite(np.asarray(a, float)))) for a in arrays)
    return Check("finite", float(bad), "-", "any", "fail" if bad else "ok")


def check_temperature(T) -> list:
    T = np.asarray(T, float)
    d = float(np.mean(T) - T_TARGET)
    return [_grade("temperature", d, lambda v: abs(v) > 2.0, lambda v: abs(v) > 5.0,
                   "|dT|>2 K", "|dT|>5 K"),
            _grade("temperature_peak", float(np.max(T)), lambda v: False, lambda v: v > 400.0,
                   "-", ">400 K")]


def check_drift(name, current_mean, previous_mean, warn_rel, fail_rel) -> Check:
    if previous_mean is None:
        return Check(name, 0.0, f">{warn_rel:.1%}", f">{fail_rel:.1%}", "ok", "first chunk")
    rel = abs(current_mean - previous_mean) / max(abs(previous_mean), 1e-12)
    return _grade(name, rel, lambda v: v > warn_rel, lambda v: v > fail_rel,
                  f">{warn_rel:.1%}", f">{fail_rel:.1%}")


def check_umbrella(cv, r0, z0) -> list:
    """cv: (n, 3) Delta x, Delta y, Delta z."""
    cv = np.asarray(cv, float)
    dc = float(np.mean(cv[:, 0]) - r0)
    lat = float(max(np.mean(np.abs(cv[:, 1])), np.mean(np.abs(cv[:, 2] - z0))))
    return [_grade("umbrella_centre", dc, lambda v: abs(v) > 0.15, lambda v: abs(v) > 0.30,
                   "|d|>0.15 nm", "|d|>0.30 nm", f"r0={r0:.3f}"),
            _grade("lateral_offset", lat, lambda v: v > 0.05, lambda v: v > 0.15,
                   ">0.05 nm", ">0.15 nm")]


# ------------------------------------------------------------- structures ---
def kabsch_rmsd(x, ref) -> float:
    """RMSD after optimal superposition (nm)."""
    a = x - x.mean(axis=0)
    b = ref - ref.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    e = (np.sum(a * a) + np.sum(b * b) - 2.0 * np.sum(s)) / len(x)
    return float(np.sqrt(max(e, 0.0)))


def count_hbonds(xyz, donors, acceptors, d_max=0.35, angle_min_deg=120.0) -> int:
    """donors: (n, 2) atom indices (N, H); acceptors: (m,) O indices; pairs on
    the same chain must already be excluded by the caller via ``allowed``."""
    N = xyz[donors[:, 0]]
    H = xyz[donors[:, 1]]
    O = xyz[acceptors]
    d = np.linalg.norm(N[:, None] - O[None], axis=-1)
    cand = np.argwhere(d < d_max)
    if cand.size == 0:
        return 0
    v1 = N[cand[:, 0]] - H[cand[:, 0]]
    v2 = O[cand[:, 1]] - H[cand[:, 0]]
    cosang = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return int(np.count_nonzero(cosang < np.cos(np.radians(angle_min_deg))))


@dataclasses.dataclass
class IntegrityRef:
    """Reference geometry of the two built segments (from system.inpcrd)."""

    backbone: list          # per segment: indices into the frame arrays
    backbone_xyz: list      # per segment: built coordinates (nm)
    hb_pairs: list          # per segment: list of (donors (n,2), acceptors per donor-chain)
    hb_ref: list            # per segment: count in the built structure
    heavy_bonds: np.ndarray  # (b, 2) heavy-atom bonds

    def to_json(self):
        return {"backbone": [list(map(int, b)) for b in self.backbone],
                "backbone_xyz": [np.asarray(x).tolist() for x in self.backbone_xyz],
                "hb_pairs": [[(d.tolist(), a.tolist()) for d, a in seg] for seg in self.hb_pairs],
                "hb_ref": [int(h) for h in self.hb_ref],
                "heavy_bonds": self.heavy_bonds.tolist()}

    @classmethod
    def from_json(cls, d):
        return cls([np.array(b) for b in d["backbone"]], [np.array(x) for x in d["backbone_xyz"]],
                   [[(np.array(dd, int).reshape(-1, 2), np.array(aa, int)) for dd, aa in seg]
                    for seg in d["hb_pairs"]], d["hb_ref"], np.array(d["heavy_bonds"], int).reshape(-1, 2))


def segment_hbonds(xyz, seg_pairs) -> int:
    return sum(count_hbonds(xyz, dn, ac) for dn, ac in seg_pairs if len(dn) and len(ac))


def build_integrity_ref(topology, xyz_nm, n_res=30) -> IntegrityRef:
    """From the subset topology the DCD frames use (solute + ions)."""
    res = [r for r in topology.residues() if r.name not in ("Na+", "Cl-", "MG", "CA", "SO4", "WAT")]
    per = n_res + 2
    chains = [res[k * per:(k + 1) * per] for k in range(6)]
    backbone, bxyz, pairs, href = [], [], [], []
    for s in range(2):
        chs = chains[3 * s:3 * s + 3]
        bb = [a.index for ch in chs for r in ch[1:-1] for a in r.atoms() if a.name in ("N", "CA", "C")]
        backbone.append(np.array(bb))
        bxyz.append(np.asarray(xyz_nm)[bb])
        seg_pairs = []
        for ci, ch in enumerate(chs):
            donors = []
            for r in ch[1:]:
                at = {a.name: a.index for a in r.atoms()}
                if "N" in at and "H" in at:
                    donors.append((at["N"], at["H"]))
            acc = [a.index for oc, other in enumerate(chs) if oc != ci for r in other
                   for a in r.atoms() if a.name == "O"]
            seg_pairs.append((np.array(donors, int).reshape(-1, 2), np.array(acc, int)))
        pairs.append(seg_pairs)
        href.append(segment_hbonds(np.asarray(xyz_nm), seg_pairs))
    heavy = [(b[0].index, b[1].index) for b in topology.bonds()
             if b[0].element is not None and b[1].element is not None
             and b[0].element.symbol != "H" and b[1].element.symbol != "H"]
    return IntegrityRef(backbone, bxyz, pairs, href, np.array(heavy, int).reshape(-1, 2))


def local_blocks(n_atoms: int, n_chain: int = 3, per_res: int = 3, block_res: int = 6):
    """Index blocks of ``block_res`` consecutive residues of all three chains,
    for backbone arrays ordered chain by chain (N, CA, C per residue)."""
    per_chain = n_atoms // n_chain
    n_res = per_chain // per_res
    out = []
    for st in range(0, n_res - block_res + 1, block_res):
        out.append(np.concatenate([np.arange(c * per_chain + per_res * st, c * per_chain + per_res * (st + block_res))
                                   for c in range(n_chain)]))
    return out


def check_structure(frames_nm, ref: IntegrityRef) -> list:
    """frames_nm: (n_frame, n_atom, 3) in the subset's atom order, made whole
    (no periodic wrapping inside a segment)."""
    frames = np.asarray(frames_nm, float)
    out = [check_finite(frames)]
    if out[0].verdict == "fail":
        return out
    b = ref.heavy_bonds
    longest = float(max(np.linalg.norm(f[b[:, 0]] - f[b[:, 1]], axis=1).max() for f in frames)) if len(b) else 0.0
    out.append(_grade("bond_stretch", longest, lambda v: v > 0.20, lambda v: v > 0.25, ">0.20 nm", ">0.25 nm"))
    glob = max(float(np.mean([kabsch_rmsd(f[idx], x0) for f in frames]))
               for idx, x0 in zip(ref.backbone, ref.backbone_xyz))
    local = max(float(np.mean([max(kabsch_rmsd(f[idx[blk]], x0[blk])
                                   for blk in (local_blocks(len(idx)) or [np.arange(len(idx))]))
                               for f in frames]))
                for idx, x0 in zip(ref.backbone, ref.backbone_xyz))
    out.append(_grade("helix_rmsd_local", local, lambda v: v > 0.20, lambda v: v > 0.30, ">0.20 nm", ">0.30 nm"))
    out.append(_grade("helix_rmsd_global", glob, lambda v: v > 0.50, lambda v: False, ">0.50 nm", "-"))
    frac = min(float(np.mean([segment_hbonds(f, sp) for f in frames])) / max(h, 1)
               for sp, h in zip(ref.hb_pairs, ref.hb_ref))
    out.append(_grade("interchain_hbonds", frac, lambda v: v < 0.6, lambda v: v < 0.4,
                      "<60% of built", "<40% of built", f"built={ref.hb_ref}"))
    return out


def check_ion_count(found: dict, expected: dict) -> Check:
    diff = {k: found.get(k, 0) - expected.get(k, 0) for k in set(found) | set(expected)
            if found.get(k, 0) != expected.get(k, 0)}
    return Check("ion_count", float(sum(abs(v) for v in diff.values())), "-", "any",
                 "fail" if diff else "ok", json.dumps(diff) if diff else "")


def coordination(xyz, ion_idx, oxygen_idx, box, r_cut) -> np.ndarray:
    if len(ion_idx) == 0:
        return np.zeros(0)
    d = xyz[oxygen_idx][None] - xyz[ion_idx][:, None]
    d -= box * np.round(d / box)
    return np.count_nonzero(np.linalg.norm(d, axis=-1) < r_cut, axis=1)


def check_ion_coordination(cn_mg, cn_ca) -> list:
    out = []
    if len(cn_mg):
        v = float(np.mean(cn_mg))
        out.append(_grade("ion_coordination_Mg", v, lambda x: not 5.5 <= x <= 6.5,
                          lambda x: not 4.0 <= x <= 8.0, "outside 5.5-6.5", "outside 4-8"))
    if len(cn_ca):
        v = float(np.mean(cn_ca))
        out.append(_grade("ion_coordination_Ca", v, lambda x: not 6.0 <= x <= 8.5,
                          lambda x: not 4.0 <= x <= 10.0, "outside 6-8.5", "outside 4-10"))
    return out


# ------------------------------------------------------------ campaign level ---
def overlap(a, b, width=0.01) -> float:
    """Overlap coefficient sum_i min(p_i, q_i) of two samples' histograms."""
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    edges = np.arange(lo, hi + width, width)
    p = np.histogram(a, edges)[0] / len(a)
    q = np.histogram(b, edges)[0] / len(b)
    return float(np.minimum(p, q).sum())


OVERLAP_WARN = 0.03
CONVERGENCE_WARN_KT = 1.0


# ------------------------------------------------------------------ logging ---
def record(checks, where: str, warnings_file: Path) -> str:
    """Append warn/fail lines; return the worst verdict."""
    worst = "ok"
    lines = []
    for c in checks:
        if c.verdict != "ok":
            lines.append(c.line(where))
        if c.verdict == "fail" or (c.verdict == "warn" and worst == "ok"):
            worst = c.verdict if worst != "fail" else worst
    if lines:
        with open(warnings_file, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    return worst
