"""Two-stage RESP fitting (Bayly, Cieplak, Cornell & Kollman, J. Phys. Chem.
97 (1993) 10269), the charge procedure of the AMBER protein force fields
(Cornell et al. 1995; ff14SB keeps those charges).

Pure numpy: the electrostatic potential comes from a quantum chemistry code
(see ``derive.py``); this module only places the fitting points and solves
the restrained least-squares problem.

Units: positions in angstrom on input, converted to bohr internally;
potentials in hartree per elementary charge; charges in e.
"""
from __future__ import annotations

import dataclasses

import numpy as np

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# Merz-Kollman radii, angstrom (Singh & Kollman, J. Comput. Chem. 5 (1984)
# 129; the values Gaussian and the psi4 RESP plugin use).
MK_RADII = {"H": 1.2, "C": 1.5, "N": 1.5, "O": 1.4, "S": 1.75, "P": 1.8}
MK_SCALES = (1.4, 1.6, 1.8, 2.0)


def sphere_points(n: int) -> np.ndarray:
    """n nearly uniform unit vectors (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    t = np.pi * (1.0 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(t), r * np.sin(t), z])


def mk_grid(xyz_A, elements, scales=MK_SCALES, density=1.0) -> np.ndarray:
    """Fitting points on the scaled van der Waals surfaces (angstrom):
    for each scale, points on every atom's sphere of radius scale * R_MK
    that lie outside all other atoms' spheres at that scale. ``density`` is
    points per square angstrom."""
    xyz = np.asarray(xyz_A, float)
    radii = np.array([MK_RADII[e] for e in elements])
    out = []
    for s in scales:
        rad = s * radii
        for i in range(len(xyz)):
            n = max(8, int(round(4.0 * np.pi * rad[i] ** 2 * density)))
            pts = xyz[i] + rad[i] * sphere_points(n)
            d = np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=-1)
            inside = (d < rad[None, :] - 1e-9)
            inside[:, i] = False
            out.append(pts[~inside.any(axis=1)])
    return np.concatenate(out)


def design_matrix(xyz_A, grid_A) -> np.ndarray:
    """A[k, i] = 1 / r_ik in bohr^-1 (potential of a unit charge at atom i
    at grid point k, hartree / e)."""
    d = np.linalg.norm(np.asarray(grid_A)[:, None, :] - np.asarray(xyz_A)[None, :, :], axis=-1)
    return 1.0 / (d * BOHR_PER_ANGSTROM)


@dataclasses.dataclass
class Conformer:
    xyz: np.ndarray       # (n_atom, 3) angstrom
    grid: np.ndarray      # (n_point, 3) angstrom
    esp: np.ndarray       # (n_point,) hartree / e


def fit(conformers, total_charge: float, groups, fixed=None, restrained=None,
        a: float = 0.0005, b: float = 0.1, q_start=None, max_iter: int = 200,
        tol: float = 1e-7):
    """Restrained ESP fit over several conformers sharing one set of charges.

    ``groups``      list of lists of atom indices; atoms in one group share a
                    charge (equivalence). Every atom must be in exactly one.
    ``fixed``       {atom index: charge} held at the given value.
    ``restrained``  boolean mask of atoms carrying the hyperbolic restraint
                    a * (sqrt(q^2 + b^2) - b) (per atom, so a group of m
                    restrained atoms is restrained m times).
    Returns (charges per atom, rrms over all points).
    """
    n_atom = len(conformers[0].xyz)
    fixed = {} if fixed is None else dict(fixed)
    restrained = np.zeros(n_atom, bool) if restrained is None else np.asarray(restrained, bool)
    seen = sorted(i for g in groups for i in g)
    if seen != list(range(n_atom)):
        raise ValueError("groups must partition the atoms")
    free_groups = [g for g in groups if not any(i in fixed for i in g)]
    for g in groups:
        vals = {fixed[i] for i in g if i in fixed}
        if len(vals) > 1 or (vals and not all(i in fixed for i in g)):
            raise ValueError(f"group {g} is partly fixed or fixed to different values")
    q_fixed = np.zeros(n_atom)
    for i, v in fixed.items():
        q_fixed[i] = v
    m = len(free_groups)
    M = np.zeros((n_atom, m))
    for j, g in enumerate(free_groups):
        M[g, j] = 1.0

    # normal equations of sum over conformers of |A q - V|^2 in the free parameters
    H = np.zeros((m, m))
    g_vec = np.zeros(m)
    ss = 0.0
    for c in conformers:
        A = design_matrix(c.xyz, c.grid)
        AM = A @ M
        resid0 = c.esp - A @ q_fixed
        H += AM.T @ AM
        g_vec += AM.T @ resid0
        ss += float(c.esp @ c.esp)
    n_restr = M[restrained].sum(axis=0)          # restrained atoms per parameter
    ones = M.sum(axis=0)                         # atoms per parameter
    q_target = total_charge - q_fixed.sum()

    p = np.zeros(m) if q_start is None else np.linalg.lstsq(M, np.asarray(q_start) - q_fixed, rcond=None)[0]
    for _ in range(max_iter):
        w = a * n_restr / np.sqrt(p * p + b * b)  # linearised restraint
        K = np.zeros((m + 1, m + 1))
        K[:m, :m] = H + np.diag(w)
        K[:m, m] = ones
        K[m, :m] = ones
        rhs = np.concatenate([g_vec, [q_target]])
        p_new = np.linalg.solve(K, rhs)[:m]
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    q = M @ p + q_fixed
    # relative rms deviation, the RESP program's quality measure
    num = sum(float(np.sum((design_matrix(c.xyz, c.grid) @ q - c.esp) ** 2)) for c in conformers)
    return q, float(np.sqrt(num / ss))


def two_stage(conformers, total_charge, elements, bonds, fixed=None, symmetric=()):
    """Standard two-stage RESP.

    Stage 1: a = 0.0005 on every heavy atom, no equivalences except
    ``symmetric`` (lists of atoms equivalent by symmetry, e.g. carboxylate
    oxygens). Stage 2: a = 0.001 on sp3 carbons bearing hydrogens (methyl and
    methylene); those carbons and their hydrogens are refitted with the
    hydrogens on each carbon equivalenced, everything else held at its
    stage-1 value.
    """
    n = len(elements)
    fixed = {} if fixed is None else dict(fixed)
    sym = [list(s) for s in symmetric]
    in_sym = {i for s in sym for i in s}
    groups1 = sym + [[i] for i in range(n) if i not in in_sym]
    heavy = np.array([e != "H" for e in elements])
    q1, rrms1 = fit(conformers, total_charge, groups1, fixed, heavy, a=0.0005)

    nbr = {i: set() for i in range(n)}
    for i, j in bonds:
        nbr[i].add(j)
        nbr[j].add(i)
    refit = set()
    groups2 = []
    for i in range(n):
        if elements[i] != "C" or len(nbr[i]) != 4 or i in fixed:
            continue
        hs = sorted(j for j in nbr[i] if elements[j] == "H" and j not in fixed)
        if len(hs) < 2:
            continue
        refit.add(i)
        refit.update(hs)
        groups2 += [[i], hs]
    if not refit:                             # no methyl/methylene groups: stage 1 is final
        return q1, {"rrms_stage1": rrms1, "rrms_stage2": rrms1, "q_stage1": q1}
    fixed2 = {i: (fixed[i] if i in fixed else q1[i]) for i in range(n) if i not in refit}
    groups2 += [[i] for i in range(n) if i not in refit]
    restr2 = np.array([i in refit and elements[i] == "C" for i in range(n)])
    # the total-charge constraint is still imposed; with everything else
    # fixed it moves only the refitted groups
    q2, rrms2 = fit(conformers, total_charge, groups2, fixed2, restr2, a=0.001, q_start=q1)
    return q2, {"rrms_stage1": rrms1, "rrms_stage2": rrms2, "q_stage1": q1}
