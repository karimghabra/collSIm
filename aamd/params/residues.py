"""Non-standard residues of the campaign: what their atoms are, how they are
built onto a lysine, and which AMBER ff14SB atom types they take.

Every atom type below already exists in ff14SB with all the bonded terms the
residue needs, because each new group is a copy of a group the force field
already parametrises:

* the N-epsilon amide of acetyl- and succinyl-lysine is a peptide bond
  (types N, H, C, O, as the backbone and the ACE/NME caps);
* the acetyl methyl is the ACE methyl (CT, HC);
* the succinyl CH2-CH2-COO(-) takes the ff94 glutamate types (CT, HC, C,
  O2); its protonated form those of GLH (C, O, OH, HO);
* the lysine CB..CE take the type 8C of the ff14SB-compatible acetyl-lysine
  that AmberTools ships (frcmod.ff14SBmodAA, Belfon & Simmerling), whose
  bonded terms include the CE-NZ(amide) bond; CE carries H1 hydrogens.

Only the charges are new; they come from RESP (``derive.py``).

Residue names: ALY is the PDB chemical component for N6-acetyl-lysine and
the AmberTools residue, used as shipped. SLL (N6-succinyl-lysine,
carboxylate) and SLH (the same, carboxylic acid) are built here; ALY is also
refitted with the same procedure, as a check of the procedure.
"""
from __future__ import annotations

import numpy as np

# Backbone charges shared by every lysine derivative here: the neutral-residue
# backbone of ff94/ff14SB (N -0.4157, H 0.2719, C 0.5972, O -0.5679), which is
# also what the ff14SB-compatible acetyl-lysine of AmberTools (ALY in
# mod_amino.lib, leaprc.protein.ff14SB_modAA, Belfon & Simmerling) uses.
BACKBONE_CHARGES = {"N": -0.4157, "H": 0.2719, "C": 0.5972, "O": -0.5679}
# ff14SB cap charges (amino12.lib ACE, NME); each cap sums to zero.
ACE_CHARGES = {"HH31": 0.1123, "CH3": -0.3662, "HH32": 0.1123, "HH33": 0.1123,
               "C": 0.5972, "O": -0.5679}
NME_CHARGES = {"N": -0.4157, "H": 0.2719, "CH3": -0.1490, "HH31": 0.0976,
               "HH32": 0.0976, "HH33": 0.0976}

# Atom types of the lysine part, as in the AmberTools ALY: 8C (sp3 carbon of
# ALY, frcmod.ff14SBmodAA) for CB..CE, which carries bonded terms to an amide N.
_LYS_PART = {"N": "N", "H": "H", "CA": "CX", "HA": "H1", "CB": "8C", "HB2": "HC",
             "HB3": "HC", "CG": "8C", "HG2": "HC", "HG3": "HC", "CD": "8C",
             "HD2": "HC", "HD3": "HC", "CE": "8C", "HE2": "H1", "HE3": "H1",
             "NZ": "N", "HZ": "H", "C": "C", "O": "O"}

RESIDUES = {
    # name: (net charge, atoms beyond NZ/HZ with ff14SB types)
    "ALY": (0, {"CH": "C", "OH": "O", "CH3": "CT", "HH31": "HC", "HH32": "HC",
                "HH33": "HC"}),
    "SLL": (-1, {"CS1": "C", "OS1": "O", "CS2": "CT", "HS21": "HC", "HS22": "HC",
                 "CS3": "CT", "HS31": "HC", "HS32": "HC", "CS4": "C", "OS2": "O2",
                 "OS3": "O2"}),
    "SLH": (0, {"CS1": "C", "OS1": "O", "CS2": "CT", "HS21": "HC", "HS22": "HC",
                "CS3": "CT", "HS31": "HC", "HS32": "HC", "CS4": "C", "OS2": "O",
                "OS3": "OH", "HS3": "HO"}),
}

# Equivalences by symmetry imposed in stage 1 (resonance-equivalent atoms).
SYMMETRIC = {"SLL": [("OS2", "OS3")], "ALY": [], "SLH": []}


def atom_types(resname: str) -> dict:
    """ff14SB type of every atom of the residue."""
    types = dict(_LYS_PART)
    types.update(RESIDUES[resname][1])
    return types


def element(name: str) -> str:
    return name[0]


# ------------------------------------------------------------ construction ---
def place(a, b, c, bond, angle_deg, torsion_deg):
    """Position of d with |cd| = bond, angle b-c-d and dihedral a-b-c-d
    (natural extension reference frame)."""
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    ang, tor = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor),
                   bond * np.sin(ang) * np.sin(tor)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def dihedral(a, b, c, d) -> float:
    """Dihedral a-b-c-d in degrees."""
    b0, b1, b2 = (np.asarray(a) - b), (np.asarray(c) - b), (np.asarray(d) - c)
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def acylate(atoms: dict, resname: str) -> dict:
    """Turn the side chain of a lysine into ``resname``.

    ``atoms`` maps atom name -> xyz (angstrom) and must hold CD, CE, NZ (any
    hydrogens of NZ are dropped). Returns a new dict with the acyl atoms
    built in standard geometry, the amide trans and the chain extended. Heavy
    atoms only if ``atoms`` has no CE hydrogens (as for the template); with
    hydrogens present, all hydrogens of the new groups are built too.
    """
    if resname == "LYS":
        return dict(atoms)
    out = {k: np.asarray(v, float) for k, v in atoms.items() if not k.startswith("HZ")}
    with_h = "HE2" in atoms
    cd, ce, nz = out["CD"], out["CE"], out["NZ"]
    c1 = place(cd, ce, nz, 1.335, 121.9, 180.0)     # amide carbon, trans to CD
    o1 = place(ce, nz, c1, 1.229, 123.0, 0.0)
    c2 = place(ce, nz, c1, 1.515, 116.0, 180.0)
    if resname == "ALY":
        out.update(CH=c1, OH=o1, CH3=c2)
        if with_h:
            out["HZ"] = place(cd, ce, nz, 1.01, 118.0, 0.0)
            for k, t in zip(("HH31", "HH32", "HH33"), (60.0, 180.0, 300.0)):
                out[k] = place(nz, c1, c2, 1.09, 109.5, t)
        return out
    c3 = place(nz, c1, c2, 1.53, 112.0, 180.0)
    c4 = place(c1, c2, c3, 1.52, 112.0, 180.0)
    o2 = place(c2, c3, c4, 1.25, 118.0, 0.0)
    o3 = place(c2, c3, c4, 1.25, 118.0, 180.0)
    out.update(CS1=c1, OS1=o1, CS2=c2, CS3=c3, CS4=c4, OS2=o2, OS3=o3)
    if with_h:
        out["HZ"] = place(cd, ce, nz, 1.01, 118.0, 0.0)
        out["HS21"] = place(nz, c1, c2, 1.09, 109.5, 60.0)
        out["HS22"] = place(nz, c1, c2, 1.09, 109.5, -60.0)
        out["HS31"] = place(c1, c2, c3, 1.09, 109.5, 60.0)
        out["HS32"] = place(c1, c2, c3, 1.09, 109.5, -60.0)
        if resname == "SLH":
            out["OS2"] = place(c2, c3, c4, 1.21, 123.0, 0.0)
            out["OS3"] = place(c2, c3, c4, 1.34, 113.0, 180.0)
            out["HS3"] = place(c3, c4, out["OS3"], 0.97, 108.0, 180.0)
    return out
