"""OpenMM set-up: force field terms beyond the prmtop, restraints, umbrella
windows.

Force field (``make_system``)
-----------------------------
The prmtop from ``aamd.build`` carries ff14SB, TIP3P, Joung-Cheatham Na+/Cl-
and the 12-6 part of the Li-Merz 12-6-4 Mg2+/Ca2+ set. The r^-4 term of
12-6-4 is added here as a CustomNonbondedForce between the divalent cations
and every other atom, with exactly the combination parmed's ``add12_6_4``
writes into LENNARD_JONES_CCOEF:

    C4(M, j) = C4_M(water) * alpha_j / alpha_OW,    E = -C4 / r^4

alpha_j from AmberTools' lj_1264_pol.dat by atom type (alpha_OW = 1.444 A^3,
so a water oxygen gets C4_M(water) itself), C4_M(water) from parmed's TIP3P
table (Mg 132.9, Ca 87.3 kcal/mol/A^4, Li & Merz 2014). For the carboxylate
oxygen type O2 the polarizability is replaced by the value Jafari et al.
(J. Phys. Chem. B 128 (2024) 684) fitted to the experimental free energy of
Mg2+ and Ca2+ binding to acetate in TIP3P with ff14SB: 0.125 A^3 for Mg2+ and
0.600 A^3 for Ca2+ (C4 = 11.50 and 36.27 kcal/mol/A^4). Every carboxylate in
these systems (Asp, Glu, succinyl-Lys) has O2 oxygens.

Restraints (``add_restraints``)
-------------------------------
Groups: the CA atoms of each segment (centroid), and for each segment the CA
atoms of the first and of the last triplet of every chain (its two ends).

* umbrella on the inter-axial separation, x_B - x_A of the two centroids,
  k_u (global parameters ``r0`` and ``k_umb``);
* y_B - y_A and z_B - z_A - z0 held at zero by a stiff harmonic term: the
  pair stays side by side along x (so the separation is a 1-D coordinate with
  no rotational Jacobian) at the built axial stagger;
* each segment's axis held parallel to z (x and y of its two end centroids
  equal);
* the helix ends held intact: an elastic network of harmonic distance
  restraints between the backbone N, CA, C atoms of the two terminal
  triplets at each end of each segment (pairs closer than 0.8 nm in the
  built ideal helix, 1-2 and 1-3 neighbours excluded), at their built
  lengths. It holds the ends' geometry without fixing them in space. (An
  RMSD collective variable does the same but costs a factor of four in
  speed in OpenMM.)

Nothing restrains the rotation of either segment about its own axis.

Units: nm, ps, kJ/mol inside OpenMM.
"""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

KCAL_A4_TO_KJ_NM4 = 4.184e-4
ALPHA_WATER = 1.444
C4_WATER_TIP3P = {"Mg": 132.9, "Ca": 87.3}          # kcal/mol/A^4, parmed DEFAULT_C4_PARAMS
JAFARI_O2_POL = {"Mg": 0.125, "Ca": 0.600}          # A^3, acetate-fitted (Jafari et al. 2024)
DIVALENT_RESNAMES = {"MG": "Mg", "CA": "Ca"}
TEMPERATURE = 298.15


def polarizabilities():
    """{AMBER atom type: alpha (A^3)} from AmberTools' lj_1264_pol.dat."""
    amberhome = Path(shutil.which("tleap")).resolve().parent.parent
    pol = {}
    for line in open(amberhome / "dat" / "leap" / "parm" / "lj_1264_pol.dat"):
        t = line.split()
        if len(t) >= 2:
            try:
                pol[t[0]] = float(t[1])
            except ValueError:
                pass
    return pol


def c4_matrix_terms(atom_types, resnames, pol=None, o2_override=True):
    """Per-atom (c4_water, alpha) for the r^-4 term. c4_water is non-zero only
    on divalent cations (kcal/mol/A^4); alpha in A^3 (0 for the cations, whose
    mutual r^-4 term parmed does not create unless their type has an entry)."""
    pol = polarizabilities() if pol is None else pol
    c4w = np.zeros(len(atom_types))
    alpha = np.zeros(len(atom_types))
    cations = {DIVALENT_RESNAMES[r] for r in resnames if r in DIVALENT_RESNAMES}
    if len(cations) > 1:
        raise ValueError("one divalent cation species per system (O2 polarizability is per species)")
    cation = next(iter(cations), None)
    for i, (t, r) in enumerate(zip(atom_types, resnames)):
        if r in DIVALENT_RESNAMES:
            c4w[i] = C4_WATER_TIP3P[DIVALENT_RESNAMES[r]]
            continue
        a = pol.get(t)
        if a is None:
            raise KeyError(f"no 12-6-4 polarizability for atom type {t}")
        if t == "O2" and cation is not None and o2_override:
            a = JAFARI_O2_POL[cation]
        alpha[i] = a
    return c4w, alpha


def add_c4_term(system, prmtop, cutoff_nm: float, o2_override=True):
    """Add E = -sum_{M,j} C4(M,j) / r^4 (M divalent cation)."""
    atoms = list(prmtop.topology.atoms())
    types = list(prmtop._prmtop._raw_data["AMBER_ATOM_TYPE"])
    resn = [a.residue.name for a in atoms]
    c4w, alpha = c4_matrix_terms(types, resn, o2_override=o2_override)
    ions = np.flatnonzero(c4w > 0)
    if ions.size == 0:
        return None
    f = openmm.CustomNonbondedForce(
        "-(c4w1*alpha2 + c4w2*alpha1)*scale/r^4")
    f.addGlobalParameter("scale", KCAL_A4_TO_KJ_NM4 / ALPHA_WATER)
    f.addPerParticleParameter("c4w")
    f.addPerParticleParameter("alpha")
    for i in range(len(atoms)):
        f.addParticle([float(c4w[i]), float(alpha[i])])
    f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(cutoff_nm)
    f.setUseSwitchingFunction(True)
    f.setSwitchingDistance(cutoff_nm - 0.1)
    f.setUseLongRangeCorrection(False)
    others = np.setdiff1d(np.arange(len(atoms)), ions)
    f.addInteractionGroup([int(i) for i in ions], [int(i) for i in others])
    # the CUDA platform requires identical exclusions in all nonbonded forces
    nb = next(x for x in system.getForces() if isinstance(x, openmm.NonbondedForce))
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        f.addExclusion(i, j)
    system.addForce(f)
    return f


def make_system(prmtop, cutoff_nm=0.9, hmr_amu=3.0, o2_override=True):
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm * unit.nanometer,
                                 constraints=app.HBonds, rigidWater=True,
                                 hydrogenMass=hmr_amu * unit.amu, ewaldErrorTolerance=5e-4)
    add_c4_term(system, prmtop, cutoff_nm, o2_override=o2_override)
    return system


# ----------------------------------------------------------------- groups ---
@dataclasses.dataclass
class Groups:
    """Atom index groups of the two segments."""

    ca: tuple            # (CA of segment A, CA of segment B), non-cap residues
    ends: tuple          # ((A lo, A hi), (B lo, B hi)): CA of the terminal triplets
    rmsd_ends: list      # backbone N, CA, C of the two terminal triplets, per end
    heavy: tuple         # all heavy atoms of each segment
    solute: np.ndarray   # every atom of both segments


def find_groups(topology, n_res=30) -> Groups:
    chains = [c for c in topology.chains()]
    prot = [c for c in chains if any(r.name not in ("WAT", "HOH", "Na+", "Cl-", "NA", "CL", "MG", "CA", "SO4")
                                     for r in c.residues())]
    # tleap writes one chain per TER; the first three protein chains are A
    residues = [r for r in topology.residues() if r.name not in ("WAT", "HOH", "Na+", "Cl-", "NA", "CL",
                                                                 "MG", "CA", "SO4")]
    per_chain = n_res + 2
    if len(residues) != 6 * per_chain:
        raise ValueError(f"expected {6 * per_chain} solute residues, found {len(residues)}")
    seg_chains = [residues[k * per_chain:(k + 1) * per_chain] for k in range(6)]
    del prot

    def atoms(res_list, names):
        return [a.index for r in res_list for a in r.atoms() if a.name in names]

    ca, ends, rmsd_ends, heavy = [], [], [], []
    for s in range(2):
        chs = seg_chains[3 * s:3 * s + 3]
        body = [r for ch in chs for r in ch[1:-1]]
        ca.append(atoms(body, ("CA",)))
        lo = atoms([r for ch in chs for r in ch[1:4]], ("CA",))
        hi = atoms([r for ch in chs for r in ch[-4:-1]], ("CA",))
        ends.append((lo, hi))
        rmsd_ends.append(atoms([r for ch in chs for r in ch[1:7]], ("N", "CA", "C")))
        rmsd_ends.append(atoms([r for ch in chs for r in ch[-7:-1]], ("N", "CA", "C")))
        heavy.append([a.index for ch in chs for r in ch for a in r.atoms()
                      if a.element is not None and a.element.symbol != "H"])
    solute = np.array([a.index for ch in seg_chains for r in ch for a in r.atoms()])
    return Groups(tuple(ca), tuple(ends), rmsd_ends, tuple(heavy), solute)


@dataclasses.dataclass(frozen=True)
class RestraintSpec:
    k_umb: float = 2000.0       # kJ/mol/nm^2 on the separation
    k_perp: float = 5000.0      # kJ/mol/nm^2 on Delta y and Delta z
    k_align: float = 1000.0     # kJ/mol/nm^2 on the end-centroid xy offset
    k_end: float = 2000.0       # kJ/mol/nm^2 on each end distance
    end_cutoff_nm: float = 0.8  # pairs of end backbone atoms closer than this are restrained


def end_pairs(pos_nm, idx, cutoff_nm: float = RestraintSpec.end_cutoff_nm):
    """Pairs of backbone atoms of one helix end, on different chains or more
    than two residues apart in a chain, closer than ``cutoff_nm`` in the
    built helix: an elastic network that holds the end's triple-helical
    geometry without fixing it in space."""
    idx = np.asarray(idx)
    p = np.asarray(pos_nm)[idx]
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    i, j = np.nonzero(np.triu(d < cutoff_nm, k=1))
    return [(idx[a], idx[b]) for a, b in zip(i, j) if d[a, b] > 0.25]


def add_restraints(system, groups: Groups, positions_nm, spec: RestraintSpec = RestraintSpec(),
                   r0=2.0):
    """Add the restraints of the module docstring; returns the umbrella force."""
    pos = np.asarray(positions_nm)
    ca_a, ca_b = groups.ca
    z0 = float(pos[ca_b, 2].mean() - pos[ca_a, 2].mean())
    umb = openmm.CustomCentroidBondForce(
        2, "0.5*k_umb*(x2-x1-r0)^2 + 0.5*k_perp*((y2-y1)^2 + (z2-z1-z0)^2)")
    umb.addGlobalParameter("k_umb", spec.k_umb)
    umb.addGlobalParameter("r0", r0)
    umb.addGlobalParameter("k_perp", spec.k_perp)
    umb.addGlobalParameter("z0", z0)
    umb.addGroup([int(i) for i in ca_a])
    umb.addGroup([int(i) for i in ca_b])
    umb.addBond([0, 1], [])
    umb.setUsesPeriodicBoundaryConditions(False)
    umb.setForceGroup(1)
    system.addForce(umb)

    al = openmm.CustomCentroidBondForce(2, "0.5*k_align*((x2-x1)^2 + (y2-y1)^2)")
    al.addGlobalParameter("k_align", spec.k_align)
    for lo, hi in groups.ends:
        g0 = al.addGroup([int(i) for i in lo])
        g1 = al.addGroup([int(i) for i in hi])
        al.addBond([g0, g1], [])
    al.setUsesPeriodicBoundaryConditions(False)
    al.setForceGroup(2)
    system.addForce(al)

    ends = openmm.CustomBondForce("0.5*k_end*(r-r_ref)^2")
    ends.addGlobalParameter("k_end", spec.k_end)
    ends.addPerBondParameter("r_ref")
    for idx in groups.rmsd_ends:
        for a, b in end_pairs(pos, idx):
            ends.addBond(int(a), int(b), [float(np.linalg.norm(pos[a] - pos[b]))])
    ends.setUsesPeriodicBoundaryConditions(False)
    ends.setForceGroup(2)
    system.addForce(ends)
    return umb


# -------------------------------------------------------------- observables ---
def centroid_offsets(pos_nm, groups: Groups):
    """(Delta x, Delta y, Delta z) of the CA centroids, B minus A (nm)."""
    a = pos_nm[groups.ca[0]].mean(axis=0)
    b = pos_nm[groups.ca[1]].mean(axis=0)
    return b - a


def make_simulation(prmtop_path, inpcrd_path, platform="CUDA", r0=2.0, restraint=RestraintSpec(),
                    dt_fs=4.0, barostat=True, threads=4, o2_override=True):
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
    system = make_system(prmtop, o2_override=o2_override)
    groups = find_groups(prmtop.topology)
    pos = inpcrd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    add_restraints(system, groups, pos, restraint, r0=r0)
    if barostat:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, TEMPERATURE * unit.kelvin, 25))
    integ = openmm.LangevinMiddleIntegrator(TEMPERATURE * unit.kelvin, 1.0 / unit.picosecond,
                                            dt_fs * unit.femtosecond)
    plat = openmm.Platform.getPlatformByName(platform)
    # blocking sync: the host thread sleeps while the GPU works instead of
    # spinning a core, which leaves the CPU to other jobs at a few per cent of speed
    props = {"Precision": "mixed", "UseBlockingSync": "true"} if platform == "CUDA" else ({"Threads": str(threads)} if platform == "CPU" else {})
    sim = app.Simulation(prmtop.topology, system, integ, plat, props)
    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    return sim, groups, prmtop
