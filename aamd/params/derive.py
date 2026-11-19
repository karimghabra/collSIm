"""Derive charges and tleap libraries for the non-standard residues.

``python -m aamd.params.derive [ALY SLL SLH SO4] [--threads 4]``

Procedure (the AMBER protein standard: Cornell et al. 1995, as used for
ff94 and kept by ff14SB; the same as Belfon & Simmerling's ff14SB-compatible
non-canonical residues):

1. Capped residue ACE-X-NME, built from tleap's ACE-LYS-NME by acylating the
   lysine side chain (``residues.acylate``), in two backbone conformations,
   alpha (phi, psi) = (-60, -45) and beta (-120, 130), side chain extended
   (chi1-chi4 and the acyl backbone 180 deg).
2. Geometry: ff14SB energy minimisation in vacuum with phi, psi and the
   side-chain dihedrals restrained (provisional charges). The AMBER standard
   optimises at HF/6-31G* instead; on this machine, shared with other jobs, a
   single HF/6-31G* optimisation step took about ten minutes, so the geometry
   comes from the force field the charges are for. The check of the whole
   procedure is the refit of the AmberTools ALY (``ALY_refit.json``).
3. Electrostatic potential at HF/6-31G* on Merz-Kollman shells (1.4, 1.6, 1.8,
   2.0 x vdW radius, 1 point per A^2), ``resp.mk_grid``.
4. Two-stage RESP over both conformers together (``resp.two_stage``), with
   the cap atoms fixed at their ff14SB charges, the backbone N, H, C, O fixed
   at the ff14SB neutral-backbone values, and the residue's net charge
   imposed.
5. tleap library (``X.lib``) with the ff14SB atom types of ``residues.py``;
   any bonded term missing from ff14SB + ff14SB_modAA is reported by parmchk2
   into ``X.frcmod``.

Sulfate (SO4 2-): S-O 1.49 A tetrahedron, HF/6-31G* optimisation, RESP with the
four oxygens equivalent; GAFF2 types s6 / o (Lennard-Jones and bonded terms).

Psi4 uses density fitting for the SCF (scf_type df); the effect on a fitted
charge is below 0.001 e.

Outputs land in ``aamd/params/``: X.lib, X.frcmod, X_resp.json (charges, the
fit quality, geometries) and, for the check, ALY_refit.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import residues as R
from . import resp

HERE = Path(__file__).resolve().parent
CONFORMERS = {"alpha": (-60.0, -45.0), "beta": (-120.0, 130.0)}


# --------------------------------------------------------------- libraries ---
def amberhome() -> Path:
    return Path(shutil.which("tleap")).resolve().parent.parent


def lib_charges(resname: str, libs=("amino12.lib", "aminoct12.lib", "aminont12.lib", "mod_amino.lib")):
    """{atom: (type, charge)} of a residue in the AmberTools libraries."""
    for lib in libs:
        txt = (amberhome() / "dat" / "leap" / "lib" / lib).read_text()
        m = re.search(r"!entry\.%s\.unit\.atoms table.*?\n(.*?)\n!entry" % re.escape(resname), txt, re.S)
        if m:
            out = {}
            for line in m.group(1).strip().splitlines():
                t = line.split()
                out[t[0].strip('"')] = (t[1].strip('"'), float(t[7]))
            return out
    raise KeyError(resname)


def run_tleap(script: str, cwd: Path) -> str:
    env = dict(os.environ, AMBERHOME=str(amberhome()))
    (cwd / "leap.in").write_text(script + "\nquit\n")
    p = subprocess.run(["tleap", "-f", "leap.in"], cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0 or "FATAL" in p.stdout:
        raise RuntimeError(p.stdout[-3000:])
    return p.stdout


def read_pdb_atoms(path):
    """List of (resname, resnum, atom name, xyz)."""
    out = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            out.append((line[17:20].strip(), int(line[22:26]), line[12:16].strip(),
                        np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return out


# ---------------------------------------------------------------- molecule ---
class Capped:
    """ACE-X-NME as parallel lists (residue tag 'ACE'/'X'/'NME', name, element, xyz)."""

    def __init__(self, rows):
        self.tag = [r[0] for r in rows]
        self.name = [r[1] for r in rows]
        self.xyz = np.array([r[2] for r in rows], float)

    @property
    def elements(self):
        return [n[0] if n[0] != "C" or n[:2] != "CL" else "Cl" for n in self.name]

    def index(self, tag, name):
        for i, (t, n) in enumerate(zip(self.tag, self.name)):
            if t == tag and n == name:
                return i
        raise KeyError((tag, name))

    def bonds(self):
        el = self.elements
        rcov = {"H": 0.32, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
        d = np.linalg.norm(self.xyz[:, None] - self.xyz[None], axis=-1)
        out = []
        for i in range(len(el)):
            for j in range(i + 1, len(el)):
                if d[i, j] < 1.25 * (rcov[el[i]] + rcov[el[j]]):
                    out.append((i, j))
        return out


def capped_residue(resname: str, workdir: Path) -> Capped:
    run_tleap("source leaprc.protein.ff14SB\nm = sequence {ACE LYS NME}\nsavepdb m cap.pdb", workdir)
    rows = read_pdb_atoms(workdir / "cap.pdb")
    lys = {n: x for r, k, n, x in rows if r == "LYS"}
    new = R.acylate(lys, resname)
    out = [("ACE", n, x) for r, k, n, x in rows if r == "ACE"]
    order = [n for r, k, n, x in rows if r == "LYS" and n in new] + [n for n in new if n not in lys]
    out += [("X", n, new[n]) for n in order]
    out += [("NME", n, x) for r, k, n, x in rows if r == "NME"]
    mol = Capped(out)
    expect = set(R.atom_types(resname))
    have = {n for t, n in zip(mol.tag, mol.name) if t == "X"}
    if have != expect:
        raise ValueError(f"{resname}: built atoms {sorted(have ^ expect)} differ from the type table")
    return mol


def _subtree(bonds, n, j, k):
    """Atoms on k's side of the j-k bond."""
    nbr = {i: set() for i in range(n)}
    for a, b in bonds:
        nbr[a].add(b)
        nbr[b].add(a)
    seen, stack = {k}, [k]
    while stack:
        x = stack.pop()
        for y in nbr[x]:
            if y not in seen and y != j:
                seen.add(y)
                stack.append(y)
    if j in seen:
        raise ValueError("bond is in a ring")
    return sorted(seen)


def set_dihedral(mol: Capped, quad, value_deg, bonds):
    i, j, k, l = quad
    cur = R.dihedral(*mol.xyz[[i, j, k, l]])
    ang = np.radians(value_deg - cur)
    axis = mol.xyz[k] - mol.xyz[j]
    axis /= np.linalg.norm(axis)
    move = _subtree(bonds, len(mol.name), j, k)
    p = mol.xyz[move] - mol.xyz[k]
    c, s = np.cos(ang), np.sin(ang)
    rot = (p * c + np.cross(axis, p) * s + np.outer(p @ axis, axis) * (1 - c))
    mol.xyz[move] = rot + mol.xyz[k]


def dihedral_set(mol: Capped, resname: str, phi, psi):
    q = lambda *names: tuple(mol.index(t, n) for t, n in names)
    d = [(q(("ACE", "C"), ("X", "N"), ("X", "CA"), ("X", "C")), phi),
         (q(("X", "N"), ("X", "CA"), ("X", "C"), ("NME", "N")), psi),
         (q(("X", "N"), ("X", "CA"), ("X", "CB"), ("X", "CG")), 180.0),
         (q(("X", "CA"), ("X", "CB"), ("X", "CG"), ("X", "CD")), 180.0),
         (q(("X", "CB"), ("X", "CG"), ("X", "CD"), ("X", "CE")), 180.0),
         (q(("X", "CG"), ("X", "CD"), ("X", "CE"), ("X", "NZ")), 180.0)]
    if resname in ("SLL", "SLH"):
        d += [(q(("X", "CE"), ("X", "NZ"), ("X", "CS1"), ("X", "CS2")), 180.0),
              (q(("X", "NZ"), ("X", "CS1"), ("X", "CS2"), ("X", "CS3")), 180.0),
              (q(("X", "CS1"), ("X", "CS2"), ("X", "CS3"), ("X", "CS4")), 180.0)]
    elif resname == "ALY":
        d += [(q(("X", "CE"), ("X", "NZ"), ("X", "CH"), ("X", "CH3")), 180.0)]
    return d


# --------------------------------------------------------------------- psi4 ---
def psi4_opt_esp(elements, xyz, charge, frozen, workdir: Path, threads=4, memory="3 GB",
                 optimize=True):
    """Constrained HF/6-31G* optimisation and ESP on the MK grid.
    ``frozen`` is a list of 0-based dihedral quadruples. Returns (xyz, grid, esp)."""
    import psi4

    psi4.core.clean_options()                 # options are global in psi4: start clean every time
    psi4.core.set_output_file(str(workdir / "psi4.out"), False)
    psi4.set_num_threads(threads)
    psi4.set_memory(memory)
    geom = "\n".join(f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(elements, xyz))
    mol = psi4.geometry(f"{charge} 1\n{geom}\nsymmetry c1\nno_reorient\nno_com\nunits angstrom")
    psi4.set_options({"basis": "6-31G*", "scf_type": "df", "geom_maxiter": 50,
                      "g_convergence": "gau"})
    if frozen:
        psi4.set_options({"frozen_dihedral": " ".join(f"{a+1} {b+1} {c+1} {d+1}" for a, b, c, d in frozen)})
    if optimize:
        try:
            psi4.optimize("scf", molecule=mol)
        except psi4.OptimizationConvergenceError as ex:
            # keep the last geometry: with the dihedrals frozen and 50 steps the
            # remaining forces are small; the report records it
            mol.set_geometry(ex.wfn.molecule().geometry())
            (workdir / "not_converged").write_text("geometry optimisation stopped at geom_maxiter")
    xyz_opt = np.asarray(mol.geometry()) * 0.529177210903
    grid = resp.mk_grid(xyz_opt, elements)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        np.savetxt("grid.dat", grid, fmt="%.8f")
        e, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
        psi4.oeprop(wfn, "GRID_ESP")
        esp = np.loadtxt("grid_esp.dat")
    finally:
        os.chdir(cwd)
    psi4.core.clean()
    return xyz_opt, grid, esp


# --------------------------------------------------------------------- fit ---
def fit_residue(resname: str, confs, mol: Capped, total=None):
    total = R.RESIDUES[resname][0] if total is None else total
    ace, nme = lib_charges("ACE"), lib_charges("NME")
    fixed = {}
    for i, (t, n) in enumerate(zip(mol.tag, mol.name)):
        if t == "ACE":
            fixed[i] = ace[n][1]
        elif t == "NME":
            fixed[i] = nme[n][1]
        elif n in R.BACKBONE_CHARGES:
            fixed[i] = R.BACKBONE_CHARGES[n]
    sym = [[mol.index("X", a) for a in grp] for grp in R.SYMMETRIC.get(resname, [])]
    q, info = resp.two_stage(confs, float(total), mol.elements, mol.bonds(), fixed=fixed,
                             symmetric=sym)
    charges = {n: float(q[i]) for i, (t, n) in enumerate(zip(mol.tag, mol.name)) if t == "X"}
    # round to 1e-4 e and put the rounding residue on CA, as the AMBER libraries do
    rounded = {n: round(v, 4) for n, v in charges.items()}
    rounded["CA"] = round(rounded["CA"] + (total - sum(rounded.values())), 4)
    return rounded, info


def write_lib(resname: str, charges: dict, mol: Capped, outdir: Path):
    """tleap OFF library via a mol2 of the bare residue."""
    types = R.atom_types(resname)
    idx = [i for i, t in enumerate(mol.tag) if t == "X"]
    names = [mol.name[i] for i in idx]
    xyz = mol.xyz[idx]
    local = {i: k for k, i in enumerate(idx)}
    bonds = [(local[a], local[b]) for a, b in mol.bonds() if a in local and b in local]
    lines = ["@<TRIPOS>MOLECULE", resname, f"{len(names)} {len(bonds)} 1 0 0", "PROTEIN",
             "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for k, (n, x) in enumerate(zip(names, xyz), 1):
        lines.append(f"{k:6d} {n:<5s} {x[0]:10.4f} {x[1]:10.4f} {x[2]:10.4f} {types[n]:<4s} 1 {resname} "
                     f"{charges[n]:9.4f}")
    lines.append("@<TRIPOS>BOND")
    for k, (a, b) in enumerate(bonds, 1):
        lines.append(f"{k:6d} {a+1:5d} {b+1:5d} 1")
    lines += ["@<TRIPOS>SUBSTRUCTURE", f"     1 {resname}  1 RESIDUE 0 **** 0 ROOT", ""]
    (outdir / f"{resname}.mol2").write_text("\n".join(lines))
    run_tleap(f"source leaprc.protein.ff14SB\nsource leaprc.protein.ff14SB_modAA\n"
              f"{resname} = loadmol2 {resname}.mol2\n"
              f"set {resname} head {resname}.1.N\nset {resname} tail {resname}.1.C\n"
              f"set {resname}.1 connect0 {resname}.1.N\nset {resname}.1 connect1 {resname}.1.C\n"
              f"set {resname} restype protein\nset {resname}.1 restype protein\n"
              f"set {resname}.1 name \"{resname}\"\n"
              f"saveoff {resname} {resname}.lib", outdir)
    # parameter check on a capped peptide: tleap fails if a term is missing
    log = run_tleap(f"source leaprc.protein.ff14SB\nsource leaprc.protein.ff14SB_modAA\n"
                    f"loadoff {resname}.lib\nm = sequence {{ACE {resname} NME}}\ncheck m\n"
                    f"saveamberparm m {resname}_check.prmtop {resname}_check.inpcrd", outdir)
    return log


def provisional_charges(resname: str) -> dict:
    """Rough charges used only to minimise the geometry (the dihedrals are
    restrained): the AmberTools ALY for the lysine and amide part, glutamate-
    like values for the succinyl tail."""
    aly = {("HA" if n == "HCA" else n): c for n, (t, c) in lib_charges("ALY").items()}
    if resname == "ALY":
        return aly
    q = {n: aly[n] for n in R.atom_types(resname) if n in aly}
    q.update(CS1=aly["CH"], OS1=aly["OH"], CS2=-0.2, HS21=0.06, HS22=0.06, CS3=-0.1, HS31=0.05, HS32=0.05)
    if resname == "SLL":
        q.update(CS4=0.80, OS2=-0.82, OS3=-0.82)
    else:
        q.update(CS4=0.68, OS2=-0.58, OS3=-0.65, HS3=0.46)
    return q


def mm_minimize(mol: Capped, resname: str, quads, wd: Path, k_kj_rad2: float = 2000.0):
    """ff14SB energy minimisation of ACE-X-NME in vacuum with the listed
    dihedrals held at their targets (harmonic, k = 2000 kJ/mol/rad^2);
    updates ``mol.xyz`` in place. Provisional charges for X."""
    import openmm
    from openmm import app, unit

    if resname == "ALY":
        run_tleap("source leaprc.protein.ff14SB\nsource leaprc.protein.ff14SB_modAA\n"
                  "m = sequence {ACE ALY NME}\nsaveamberparm m cap.prmtop cap.inpcrd", wd)
    else:
        write_lib(resname, provisional_charges(resname), mol, wd)
        os.replace(wd / f"{resname}_check.prmtop", wd / "cap.prmtop")
    prm = app.AmberPrmtopFile(str(wd / "cap.prmtop"))
    tag_of = {"ACE": "ACE", "NME": "NME"}
    order = []
    for a in prm.topology.atoms():
        t = tag_of.get(a.residue.name, "X")
        n = "HA" if (t == "X" and a.name == "HCA") else a.name
        order.append(mol.index(t, n))
    system = prm.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    tor = openmm.CustomTorsionForce("0.5*k*d^2; d = min(abs(theta-t0), 2*pi-abs(theta-t0)); pi = 3.141592653589793")
    tor.addGlobalParameter("k", k_kj_rad2)
    tor.addPerTorsionParameter("t0")
    inv = {m: i for i, m in enumerate(order)}
    for quad, val in quads:
        tor.addTorsion(*[inv[q] for q in quad], [np.radians(val)])
    system.addForce(tor)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(mol.xyz[order] * 0.1)
    openmm.LocalEnergyMinimizer.minimize(ctx, 0.05, 20000)
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    mol.xyz[order] = x


def derive_residue(resname: str, outdir: Path, threads=4, refit_of=None):
    tmp = Path(tempfile.mkdtemp(prefix=f"resp_{resname}_", dir=outdir))
    mol0 = capped_residue(resname, tmp)
    bonds = mol0.bonds()
    confs, geoms = [], {}
    for tag, (phi, psi) in CONFORMERS.items():
        mol = Capped(list(zip(mol0.tag, mol0.name, mol0.xyz.copy())))
        quads = dihedral_set(mol, resname, phi, psi)
        for quad, val in quads:
            set_dihedral(mol, quad, val, bonds)
        wd = tmp / tag
        wd.mkdir()
        charge = R.RESIDUES[resname][0]
        mm_minimize(mol, resname, quads, wd)
        xyz, grid, esp = psi4_opt_esp(mol.elements, mol.xyz, charge, [q for q, _ in quads], wd, threads,
                                      optimize=False)
        confs.append(resp.Conformer(xyz, grid, esp))
        geoms[tag] = {"xyz": xyz.tolist(), "n_points": int(len(grid)),
                      "dihedrals": [R.dihedral(*xyz[list(q)]) for q, _ in quads]}
        mol0_xyz = xyz
    mol_final = Capped(list(zip(mol0.tag, mol0.name, mol0_xyz)))
    charges, info = fit_residue(resname, confs, mol_final)
    rec = {"residue": resname, "charges": charges, "rrms_stage1": info["rrms_stage1"],
           "rrms_stage2": info["rrms_stage2"], "conformers": geoms,
           "method": "two-stage RESP, ESP at HF/6-31G* (psi4, DF) on ff14SB-minimised geometries "
                     "(dihedrals restrained), MK shells 1.4-2.0, alpha+beta, caps and backbone N/H/C/O fixed"}
    name = resname if refit_of is None else f"{resname}_refit"
    if refit_of is None:
        write_lib(resname, charges, mol_final, outdir)
    else:
        ref = {n: c for n, (t, c) in lib_charges(resname).items()}
        ref = {("HA" if n == "HCA" else n): c for n, c in ref.items()}
        diff = {n: charges[n] - ref[n] for n in charges if n in ref}
        rec["reference"] = ref
        rec["difference"] = diff
        rec["rms_difference"] = float(np.sqrt(np.mean(np.square(list(diff.values())))))
        rec["max_abs_difference"] = float(np.max(np.abs(list(diff.values()))))
    (outdir / f"{name}_resp.json").write_text(json.dumps(rec, indent=1))
    return rec


def derive_sulfate(outdir: Path, threads=4):
    tmp = Path(tempfile.mkdtemp(prefix="resp_SO4_", dir=outdir))
    d = 1.49
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
    xyz = np.vstack([[0, 0, 0], d * t])
    el = ["S", "O", "O", "O", "O"]
    xyz, grid, esp = psi4_opt_esp(el, xyz, -2, [], tmp, threads)
    bonds = [(0, k) for k in range(1, 5)]
    q, info = resp.two_stage([resp.Conformer(xyz, grid, esp)], -2.0, el, bonds,
                             symmetric=[[1, 2, 3, 4]])
    qs = round(float(q[0]), 4)
    qo = round((-2.0 - qs) / 4.0, 4)
    rec = {"residue": "SO4", "charges": {"S": qs, "O1": qo, "O2": qo, "O3": qo, "O4": qo},
           "S_O_nm": float(np.linalg.norm(xyz[1] - xyz[0]) / 10.0),
           "rrms_stage1": info["rrms_stage1"], "method": "RESP HF/6-31G* (psi4), Td, O equivalent"}
    (outdir / "SO4_resp.json").write_text(json.dumps(rec, indent=1))
    lines = ["@<TRIPOS>MOLECULE", "SO4", "5 4 1 0 0", "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    names = ["S", "O1", "O2", "O3", "O4"]
    types = ["s6", "o", "o", "o", "o"]
    for k in range(5):
        c = qs if k == 0 else qo
        lines.append(f"{k+1:6d} {names[k]:<4s} {xyz[k,0]:10.4f} {xyz[k,1]:10.4f} {xyz[k,2]:10.4f} "
                     f"{types[k]:<3s} 1 SO4 {c:9.4f}")
    lines.append("@<TRIPOS>BOND")
    lines += [f"{k:6d}     1 {k+1:5d} 1" for k in range(1, 5)]
    lines += ["@<TRIPOS>SUBSTRUCTURE", "     1 SO4  1 RESIDUE 0 **** 0 ROOT", ""]
    (outdir / "SO4.mol2").write_text("\n".join(lines))
    env = dict(os.environ, AMBERHOME=str(amberhome()))
    subprocess.run(["parmchk2", "-i", "SO4.mol2", "-f", "mol2", "-o", "SO4.frcmod", "-s", "2"],
                   cwd=outdir, env=env, check=True)
    run_tleap("source leaprc.gaff2\nloadamberparams SO4.frcmod\nSO4 = loadmol2 SO4.mol2\n"
              "saveoff SO4 SO4.lib\nsaveamberparm SO4 SO4_check.prmtop SO4_check.inpcrd", outdir)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["SLL", "SLH", "SO4", "ALY"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    for w in a.what:
        if w == "SO4":
            rec = derive_sulfate(HERE, a.threads)
        elif w == "ALY":
            rec = derive_residue("ALY", HERE, a.threads, refit_of="ALY")
            print("ALY refit vs AmberTools ALY: rms", round(rec["rms_difference"], 4), "max",
                  round(rec["max_abs_difference"], 4))
        else:
            rec = derive_residue(w, HERE, a.threads)
        print(w, json.dumps(rec["charges"]))


if __name__ == "__main__":
    main()
