"""The 12-6-4 term and the restraints of aamd.md on a small tleap system."""
import shutil
import subprocess

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
parmed = pytest.importorskip("parmed")
if shutil.which("tleap") is None:
    pytest.skip("tleap not available", allow_module_level=True)

from openmm import app, unit  # noqa: E402

from aamd import md  # noqa: E402
from aamd.build import tleap  # noqa: E402


@pytest.fixture(scope="module", params=["Ca", "Mg"])
def small(request, tmp_path_factory):
    """ACE-ASP-NME + one divalent cation + water, 12-6-4 base LJ."""
    ion = request.param
    unit_name = {"Ca": "CA", "Mg": "MG"}[ion]
    wd = tmp_path_factory.mktemp(f"small_{ion}")
    tleap("source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
          "loadamberparams frcmod.ions234lm_1264_tip3p\n"
          "m = sequence {ACE ASP NME}\n"
          f"solvatebox m TIP3PBOX 8.0\naddIonsRand m {unit_name} 1\n"
          "saveamberparm m s.prmtop s.inpcrd\n", wd)
    return ion, wd


def test_c4_coefficients_equal_parmed_add12_6_4(small):
    ion, wd = small
    pol = md.polarizabilities()
    # parmed with the same polarizabilities, O2 replaced as aamd.md does
    polfile = wd / "pol.dat"
    lines = [f"{t} {(md.JAFARI_O2_POL[ion] if t == 'O2' else a)}" for t, a in pol.items()]
    polfile.write_text("\n".join(lines) + "\n")
    p = parmed.amber.AmberParm(str(wd / "s.prmtop"))
    from parmed.tools import add12_6_4, addLJType
    # in ff14SB the carbonyl O and the carboxylate O2 share one LJ type, so a
    # per-type C4 table cannot tell them apart; aamd.md applies the
    # acetate-fitted value per atom, to O2 only. Give O2 its own LJ type here
    # so that parmed's table can express the same thing.
    addLJType(p, "@%O2").execute()
    add12_6_4(p, f":{ {'Ca': 'CA', 'Mg': 'MG'}[ion]}", watermodel="TIP3P", polfile=str(polfile)).execute()
    ccoef = p.parm_data["LENNARD_JONES_CCOEF"]
    ntypes = p.pointers["NTYPES"]
    nbidx = p.parm_data["NONBONDED_PARM_INDEX"]
    tidx = p.parm_data["ATOM_TYPE_INDEX"]
    types = p.parm_data["AMBER_ATOM_TYPE"]
    resn = [a.residue.name for a in p.atoms]
    c4w, alpha = md.c4_matrix_terms(types, resn, pol)
    m = next(i for i, r in enumerate(resn) if r in ("CA", "MG"))
    checked = set()
    for j in range(len(types)):
        if j == m or types[j] in checked:
            continue
        checked.add(types[j])
        a, b = tidx[m] - 1, tidx[j] - 1
        k = nbidx[ntypes * min(a, b) + max(a, b)] - 1
        mine = c4w[m] * alpha[j] / md.ALPHA_WATER
        assert mine == pytest.approx(ccoef[k], rel=1e-6, abs=1e-9), types[j]
    assert "O2" in checked and "OW" in checked
    # the acetate-fitted values of Jafari et al. (2024), kcal/mol/A^4
    o2 = next(j for j, t in enumerate(types) if t == "O2")
    assert c4w[m] * alpha[o2] / md.ALPHA_WATER == pytest.approx({"Ca": 36.27, "Mg": 11.50}[ion], abs=0.01)


def test_c4_force_energy_has_the_right_units(small):
    ion, wd = small
    prm = app.AmberPrmtopFile(str(wd / "s.prmtop"))
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer)
    md.add_c4_term(system, prm, 0.9)
    keep = system.getNumForces() - 1           # the force just added
    for i in reversed(range(system.getNumForces())):
        if i != keep:
            system.removeForce(i)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    crd = app.AmberInpcrdFile(str(wd / "s.inpcrd"))
    pos = crd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = np.array(crd.boxVectors.value_in_unit(unit.nanometer))
    ctx.setPeriodicBoxVectors(*crd.boxVectors)
    ctx.setPositions(pos)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    # direct sum with minimum image and the same switch
    atoms = list(prm.topology.atoms())
    types = list(prm._prmtop._raw_data["AMBER_ATOM_TYPE"])
    c4w, alpha = md.c4_matrix_terms(types, [a.residue.name for a in atoms])
    m = int(np.flatnonzero(c4w > 0)[0])
    d = pos - pos[m]
    d -= np.diag(box) * np.round(d / np.diag(box))
    r = np.linalg.norm(d, axis=1)
    sel = (r > 0) & (r < 0.9)
    x = np.clip((r[sel] - 0.8) / 0.1, 0, 1)
    sw = 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3
    ref = -np.sum(c4w[m] * alpha[sel] / md.ALPHA_WATER * md.KCAL_A4_TO_KJ_NM4 / r[sel] ** 4 * sw)
    assert e == pytest.approx(ref, rel=1e-4)
    assert e < 0


def test_end_pairs_form_an_elastic_network():
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=0.4, size=(30, 3))
    pairs = md.end_pairs(pos, np.arange(30), cutoff_nm=0.6)
    d = [np.linalg.norm(pos[a] - pos[b]) for a, b in pairs]
    assert pairs and max(d) < 0.6 and min(d) > 0.25
