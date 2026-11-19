"""Build the explicit-solvent systems: two capped triple-helical segments,
water, ions.

Pipeline (``python -m aamd.build SYSTEM``):

1. Excise helix residues [start, start + n) of all three chains from the
   full-atom template that ``tools/build_tropocollagen.py`` writes
   (``assets/tropocollagen.pdb``, heavy atoms, Hyp at Y positions), once for
   segment A and once for segment B, B's window being A's shifted by the
   stagger. Each chain is capped: ACE from the preceding residue (CA -> CH3,
   C, O) and NME from the following one (N, CA -> C, the ff14SB name), so no artificial
   termini carry charge.
2. Place A on the z axis and B at (d0, 0) with its sites moved up by the
   stagger, exactly the placement of ``aamd.coarse.pair_positions`` with both
   rotations zero.
3. Give each titratable residue the protonation state chosen by
   ``aamd.coarse.discrete_charges`` (ASP/ASH, GLU/GLH, HIE/HIP, LYS) and each
   modified lysine its residue (SLL/SLH, ALY), building the acyl heavy atoms.
4. tleap: ff14SB (+ ff14SB_modAA for ALY, + the SLL/SLH/SO4 libraries made by
   ``aamd.params``), TIP3P, Joung-Cheatham Na+/Cl-, Li-Merz 12-6-4 Mg2+/Ca2+
   (the r^-4 term is added in OpenMM, ``aamd.md``), solvateBox, ions.

Units: angstrom inside PDB/tleap, nm elsewhere.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from modcoll.sites import SiteType, native_molecule

from . import coarse
from .params import residues as resdef

ROOT = Path(__file__).resolve().parent.parent
AAMD = ROOT / "aamd"
PARAMS = AAMD / "params"
WORK = AAMD / "work"            # git-ignored: topologies, trajectories
TEMPLATE = ROOT / "assets" / "tropocollagen.pdb"

AVOGADRO = 6.02214076e23
WATER_MOLARITY = 55.4           # mol/L of pure water at 298 K


# ------------------------------------------------------------------ systems ---
@dataclasses.dataclass(frozen=True)
class SystemSpec:
    """One explicit-solvent system."""

    name: str
    start_a: int                 # helix residue index of segment A's first residue
    stagger_res: int             # B's window = A's window - stagger
    chemistry: str               # "native", "succ", "acet"
    pH: float
    salt_M: float = 0.15         # NaCl on top of the neutralising counterions
    divalent: str | None = None  # "Ca", "Mg", "SO4"
    divalent_M: float = 0.0      # concentration of the divalent salt (CaCl2, MgCl2, Na2SO4)
    n_res: int = 30
    d0_nm: float = 3.05          # axis separation as built (largest window)
    buffer_nm: tuple = (1.25, 1.25, 1.2)
    # target box (angstrom): when set, the solvation buffer is adjusted until
    # tleap's box matches it to within 0.5 A, so that systems whose solute is
    # built more extended (succinyl tails) get the same volume as the native ones
    box_A: tuple | None = None
    seed: int = 20261117         # protonation pattern

    @property
    def start_b(self) -> int:
        return self.start_a - self.stagger_res


# The matrix. Segment A = helix residues 624-653; partner at 1 D (234
# residues) = 390-419, at 0.5 D (117) = 507-536. Chosen with the coarse model
# (aamd/select_segments.py): the window with the largest predicted contrast
# between 1 D and the off-D staggers for native pairs at d = 1.65 nm.
START_A = 624
# the box tleap builds around the native 1 D pair (angstrom); the modified
# systems are given the same box (see SystemSpec.box_A)
NATIVE_BOX_A = (79.66, 50.56, 125.73)
SYSTEMS = {}
for chem, pH in (("native", 7.4), ("succ", 4.2), ("acet", 7.4)):
    for tag, st in (("1D", 234), ("05D", 117)):
        base = f"{chem}-{tag}"
        box = None if chem == "native" else NATIVE_BOX_A
        SYSTEMS[base] = SystemSpec(base, START_A, st, chem, pH, box_A=box)
        for ion in ("Ca", "Mg", "SO4"):
            SYSTEMS[f"{base}-{ion}"] = SystemSpec(f"{base}-{ion}", START_A, st, chem, pH,
                                                  divalent=ion, divalent_M=0.05, box_A=box)


# ----------------------------------------------------------------- template ---
def read_template(path=TEMPLATE):
    """{(chain 0..2, res_index): (resname, {atom: xyz angstrom})}."""
    res = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            key = ("ABC".index(line[21]), int(line[22:26]) - 1)
            name = line[12:16].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            rn = line[17:20].strip()
            res.setdefault(key, (rn, {}))[1][name] = xyz
    return res


def segment_types(spec: SystemSpec, which: str):
    """(coarse Segment with modified types, per-site discrete charges)."""
    start = spec.start_a if which == "A" else spec.start_b
    seg = coarse.segment(start, spec.n_res)
    if spec.chemistry == "succ":
        seg = coarse.succinylate(seg)
    elif spec.chemistry == "acet":
        seg = coarse.acetylate(seg)
    rng = np.random.default_rng([spec.seed, 0 if which == "A" else 1])
    q = coarse.discrete_charges(seg, spec.pH, rng=rng)
    return seg, q


def residue_name(site_type: int, q: float, template_name: str) -> str:
    st = SiteType(site_type)
    if st == SiteType.ASP:
        return "ASP" if q < 0 else "ASH"
    if st == SiteType.GLU:
        return "GLU" if q < 0 else "GLH"
    if st == SiteType.HIS:
        return "HIP" if q > 0 else "HIE"
    if st == SiteType.LYS:
        return "LYS" if q > 0 else "LYN"
    if st == SiteType.SUCC_LYS:
        return "SLL" if q < 0 else "SLH"
    if st == SiteType.ACET_LYS:
        return "ALY"
    return template_name           # GLY, PRO, HYP, ALA, ... (Tyr stays neutral)


def segment_pdb_records(template, spec: SystemSpec, which: str, xy_shift, z_shift, chain_ids):
    """PDB ATOM records (heavy atoms) of one capped segment, and a summary."""
    seg, q = segment_types(spec, which)
    start = seg.start
    names = {}
    for c, ri, st, qq in zip(seg.chain, seg.res_index, seg.site_type, q):
        names[(int(c), int(ri))] = residue_name(int(st), qq, template[(int(c), int(ri))][0])
    lines = []
    serial = 1

    def put(name, rn, ch, resnum, xyz):
        nonlocal serial
        x, y, z = xyz[0] + xy_shift[0], xyz[1] + xy_shift[1], xyz[2] + z_shift
        lines.append(f"ATOM  {serial % 100000:5d} {name:<4s} {rn:>3s} {ch}{resnum:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}\n")
        serial += 1

    for c in range(3):
        ch = chain_ids[c]
        rn_prev, at_prev = template[(c, start - 1)]
        put("CH3", "ACE", ch, 1, at_prev["CA"])
        put("C", "ACE", ch, 1, at_prev["C"])
        put("O", "ACE", ch, 1, at_prev["O"])
        for k in range(spec.n_res):
            ri = start + k
            rn_t, atoms = template[(c, ri)]
            rn = names[(c, ri)]
            if rn in ("SLL", "SLH", "ALY"):
                atoms = resdef.acylate(atoms, rn)
            for an, xyz in atoms.items():
                put(an, rn, ch, k + 2, xyz)
        rn_next, at_next = template[(c, start + spec.n_res)]
        put("N", "NME", ch, spec.n_res + 2, at_next["N"])
        put("C", "NME", ch, spec.n_res + 2, at_next["CA"])
        lines.append("TER\n")
    summary = {"start": int(start), "residues": [[names[(c, start + k)] for k in range(spec.n_res)]
                                                  for c in range(3)],
               "net_charge": float(q.sum())}
    return lines, summary


def build_pdb(spec: SystemSpec, out: Path):
    """Write the two capped segments (heavy atoms) to ``out``."""
    template = read_template()
    rise_A = native_molecule().rise_nm * 10.0
    # centre segment A's window on z = 0 (template z of the window centre)
    zc = np.mean([template[(c, spec.start_a + k)][1]["CA"][2]
                  for c in range(3) for k in range(spec.n_res)])
    la, sa = segment_pdb_records(template, spec, "A", (0.0, 0.0), -zc, "ABC")
    lb, sb = segment_pdb_records(template, spec, "B", (spec.d0_nm * 10.0, 0.0),
                                 -zc + spec.stagger_res * rise_A, "DEF")
    out.write_text("".join(la + lb) + "END\n")
    return {"A": sa, "B": sb}


# -------------------------------------------------------------------- tleap ---
def count_waters(pdb_path: Path) -> int:
    return sum(1 for line in open(pdb_path) if line.startswith("ATOM") and " O   WAT" in line)


def tleap(script: str, cwd: Path) -> str:
    amberhome = Path(shutil.which("tleap")).resolve().parent.parent
    env = dict(os.environ, AMBERHOME=str(amberhome))
    (cwd / "leap.in").write_text(script + "\nquit\n")
    proc = subprocess.run(["tleap", "-f", "leap.in"], cwd=cwd, env=env,
                          capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (cwd / "leap.out").write_text(log)
    if proc.returncode != 0 or "FATAL" in log or "Could not open" in log:
        raise RuntimeError(f"tleap failed in {cwd}:\n{log[-3000:]}")
    return log


def leap_header(spec: SystemSpec) -> str:
    lines = ["source leaprc.protein.ff14SB", "source leaprc.protein.ff14SB_modAA",
             "source leaprc.water.tip3p",
             "loadamberparams frcmod.ions234lm_1264_tip3p"]
    if spec.divalent == "SO4":
        lines.append("source leaprc.gaff2")       # sulfate types s6 / o
    for lib in ("SLL", "SLH", "SO4"):
        if (PARAMS / f"{lib}.lib").exists():
            lines += [f"loadoff {PARAMS / (lib + '.lib')}"]
            if (PARAMS / f"{lib}.frcmod").exists():
                lines += [f"loadamberparams {PARAMS / (lib + '.frcmod')}"]
    return "\n".join(lines)


DIVALENT_LEAP = {"Ca": ("CA", +2, "Cl-", 2), "Mg": ("MG", +2, "Cl-", 2),
                 "SO4": ("SO4", -2, "Na+", 2)}


def ion_totals(adds) -> dict:
    """{ion: total number} over the addIonsRand commands (an ion may appear twice)."""
    out = {}
    for ion, n in adds:
        out[ion] = out.get(ion, 0) + int(n)
    return out


def build(spec: SystemSpec, workdir: Path | None = None) -> dict:
    """Run steps 1-4; returns the build record (also written as build.json)."""
    wd = (WORK / spec.name) if workdir is None else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    seg = build_pdb(spec, wd / "segments.pdb")
    hdr = leap_header(spec)
    buf = np.array([10.0 * b for b in spec.buffer_nm])
    # pass 1: solvate only, to count waters and the solute charge (and, with a
    # target box, to find the buffer that gives it)
    for _ in range(6):
        log = tleap(f"{hdr}\nm = loadpdb segments.pdb\ncharge m\n"
                    f"solvateBox m TIP3PBOX {{{buf[0]:.2f} {buf[1]:.2f} {buf[2]:.2f}}} 0.9\n"
                    f"savepdb m pass1.pdb\n", wd)
        if spec.box_A is None:
            break
        box = next(np.array(line.split()[4:7], float) for line in log.splitlines()
                   if "Total vdw box size" in line)
        miss = np.asarray(spec.box_A) - box
        if np.all(np.abs(miss) < 0.5):
            break
        buf = np.maximum(buf + 0.5 * miss, 3.0)
    bx, by, bz = buf
    q_solute = None
    for line in log.splitlines():
        if "Total unperturbed charge" in line or "Total perturbed charge" in line:
            q_solute = float(line.split(":")[-1])
            break
    if q_solute is None:
        raise RuntimeError("tleap did not report the solute charge")
    n_wat = count_waters(wd / "pass1.pdb")
    per_M = n_wat / WATER_MOLARITY           # ions per mol/L
    n_salt = int(round(spec.salt_M * per_M))
    q_int = int(round(q_solute))
    adds = []
    n_div = 0
    q_after = q_int
    if spec.divalent:
        ion, zi, counter, nc = DIVALENT_LEAP[spec.divalent]
        n_div = int(round(spec.divalent_M * per_M))
        adds.append((ion, n_div))
        adds.append((counter, nc * n_div))
        q_after += n_div * zi + nc * n_div * (1 if counter == "Na+" else -1)
    n_na = n_salt + max(0, -q_after)
    n_cl = n_salt + max(0, q_after)
    adds += [("Na+", n_na), ("Cl-", n_cl)]
    ion_cmds = "\n".join(f"addIonsRand m {ion} {n}" for ion, n in adds if n > 0)
    tleap(f"{hdr}\nm = loadpdb segments.pdb\n"
          f"solvateBox m TIP3PBOX {{{bx:.2f} {by:.2f} {bz:.2f}}} 0.9\n"
          f"{ion_cmds}\ncharge m\nsaveamberparm m system.prmtop system.inpcrd\n"
          f"savepdb m system.pdb\n", wd)
    record = {"spec": dataclasses.asdict(spec), "segments": seg, "solute_charge": q_solute,
              "n_water_pass1": n_wat, "ions": ion_totals(adds), "n_divalent": n_div,
              "buffer_A": [float(x) for x in (bx, by, bz)]}
    (wd / "build.json").write_text(json.dumps(record, indent=1))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("systems", nargs="+", help=f"names from: {', '.join(SYSTEMS)}")
    args = ap.parse_args(argv)
    for name in args.systems:
        rec = build(SYSTEMS[name])
        print(name, "solute charge", rec["solute_charge"], "ions", rec["ions"])


if __name__ == "__main__":
    main()
