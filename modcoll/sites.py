"""Per-molecule site model of type I tropocollagen.

One molecule is an array of *sites*: every residue of the three chains
(human COL1A1 P02452 x2 + COL1A2 P08123, telopeptides included) plus the
chain termini, each with a type, an axial position, an azimuth and a radial
offset on the 7/2 triple helix.

The geometry is not re-derived here. It is read from the full-atom model
that ``tools/build_tropocollagen.py`` threads through the 1K6F screw operator
(``assets/tropocollagen.pdb``; built on demand, the directory is git-ignored)
and reduced to one point per residue exactly as
``tools/azimuthal_profile.py::load_residue_surface`` does: the centroid of the
side-chain heavy atoms (CA for glycine). Two deliberate differences from that
tool, both documented in ``docs/modified_collagen_sites.md``:

* the axial origin is the N-terminal end of the helix (``s = (z - z_min)``),
  the convention ``build_tropocollagen.py`` itself uses for ``atoms.bin`` and
  ``residues.npz``. ``azimuthal_profile.py`` reads the un-shifted PDB z, which
  puts the helix at s = -13.9 ... 276.3 nm while its telopeptides are placed
  relative to 0 ... L; here both use the same origin.
* the helix and the chain ends are anchored explicitly on the UniProt
  annotation of the mature chains (CHAINS below) rather than found as "the
  longest Gly-X-Y run" with a fixed 16/25-residue window either side. The
  defaults are the intact mature chain ends; ``telopeptides="pepsin"`` gives
  the ends left by pepsin, ``"none"`` the helix alone, and ``"tool"``
  reproduces the window of the existing tool exactly.

Units: nm, rad. Charges are *not* stored here: protonation is a function of
(site array, environment) and lives in ``modcoll.electrostatics``.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import subprocess
import sys
from enum import IntEnum
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Residues per D period (Hulmes & Miller 1973; README "D = 234 residues").
D_PERIOD_RESIDUES = 234

# Axial rise per telopeptide residue, nm ("near-extended"); same value as
# tools/azimuthal_profile.py::telopeptides (RISE = 0.28).
TELO_RISE_NM = 0.28

# Radial offset given to telopeptide and terminus sites, nm. Telopeptides are
# flexible and carried as azimuthally uniform rings (the existing tool's
# treatment); the ring radius is an assumption: the mean side-chain-centroid
# radius of non-glycine helix residues in the full-atom model (0.55 nm).
TELO_RADIUS_NM = 0.55


class SiteType(IntEnum):
    """Chemical identity of a site as far as charge and modification go."""

    OTHER = 0        # no titratable group, not modifiable
    ASP = 1
    GLU = 2
    HIS = 3
    LYS = 4          # includes hydroxylysine (treated as Lys)
    ARG = 5
    NTERM = 6        # free chain alpha-amine
    CTERM = 7        # chain alpha-carboxylate
    ASN = 8          # neutral, deamidates to ASP
    GLN = 9          # neutral, deamidates to GLU
    SUCC_LYS = 10    # N-epsilon-succinyl-lysine: amine -> carboxylate
    ACET_LYS = 11    # N-epsilon-acetyl-lysine: neutral
    SUCC_NTERM = 12  # succinylated alpha-amine: carboxylate
    ACET_NTERM = 13  # acetylated alpha-amine: neutral
    TYR = 14         # phenol, ionises only above pH 9


_AA_TO_TYPE = {"D": SiteType.ASP, "E": SiteType.GLU, "H": SiteType.HIS,
               "K": SiteType.LYS, "R": SiteType.ARG, "N": SiteType.ASN,
               "Q": SiteType.GLN, "Y": SiteType.TYR}

# Kyte & Doolittle (1982) J. Mol. Biol. 157:105 hydropathy index, the table
# the existing tools use (tools/azimuthal_profile.py AA_KD1). 'O' is
# 4-hydroxyproline, given the proline value as in that tool (KD["HYP"]).
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "O": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "HYP": "O",
}

_BACKBONE = frozenset(("N", "CA", "C", "O"))

# Heavy atoms that carry the titratable group (or, for Asn/Gln, the amide that
# deamidation turns into a carboxylate). Used by position="group".
_GROUP_ATOMS = {
    "LYS": ("NZ",), "ARG": ("CZ",), "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"), "HIS": ("ND1", "NE2"), "ASN": ("OD1", "ND2"),
    "GLN": ("OE1", "NE2"), "TYR": ("OH",),
}

@dataclasses.dataclass(frozen=True)
class ChainSpec:
    """Where the mature chain and its triple-helical domain sit in a UniProt
    sequence (1-based, inclusive), and how many copies the molecule has."""

    fasta: str
    copies: int
    mature_first: int    # first residue after N-propeptide removal
    mature_last: int     # last residue before the C-propeptide
    helix_first: int     # first Gly of the triple-helical domain


# Residues in the triple-helical domain of each chain (338 Gly-X-Y triplets).
HELIX_RESIDUES = 1014

# Human type I collagen. Chains 0 and 1 are alpha1(I), chain 2 is alpha2(I),
# the assignment tools/build_tropocollagen.py makes. Boundaries are the
# UniProt chain and region annotations: P02452 "Collagen alpha-1(I) chain"
# 162-1218 with the triple helix at 179-1192; P08123 "Collagen alpha-2(I)
# chain" 80-1119 with the triple helix at 91-1104. Both mature chains start
# with Gln, which is cyclised to pyroglutamate in tissue collagen.
#
# The helix is anchored here, not searched for. "The longest run with Gly at
# every third position" (the rule of tools/build_tropocollagen.py) is 1017
# residues long for human alpha2, because its C-telopeptide starts GGG, and
# only gives the right 1014 after truncation to the alpha1 length; for bovine
# alpha2 the same rule starts one triplet early. Another species therefore
# needs its own ChainSpec entries, not just another FASTA file.
CHAINS = (ChainSpec("P02452.fasta", 2, 162, 1218, 179),
          ChainSpec("P08123.fasta", 1, 80, 1119, 91))

# Residues removed by pepsin, per chain type: (from the N-terminal end of the
# N-telopeptide, from the C-terminal end of the C-telopeptide). Pepsin does
# not remove telopeptides completely. alpha1 N-telopeptide: preferred cut
# Tyr6-Asp7, removing pQLSYGY (Qian et al. 2017, Biosci. Biotechnol. Biochem.
# 81:514, bovine); alpha1 C-telopeptide: cuts within residues 5-9, removing
# the distal 17-21 residues (Galea et al. 2000, Protein Sci. 9:1947; 19 is
# used); alpha2 N-telopeptide: two residues; alpha2 C-telopeptide: no data,
# kept whole. These are bovine cut sites applied to the human sequence.
PEPSIN_CUTS = {"P02452.fasta": (6, 19), "P08123.fasta": (2, 0)}

# The existing tool's telopeptide window: residues before / after the longest
# Gly-X-Y run (tools/azimuthal_profile.py::telopeptides).
TOOL_NTEL, TOOL_CTEL = 16, 25

TELOPEPTIDE_MODELS = ("mature", "pepsin", "none", "tool")


@dataclasses.dataclass(frozen=True)
class Molecule:
    """Site array of one tropocollagen molecule (all arrays share one length).

    Residue rows carry a hydrophobic weight; the terminus rows (``aa`` '<' for
    an alpha-amine, '>' for an alpha-carboxylate) carry only a charge.
    """

    chain: np.ndarray        # int8: 0, 1 = alpha1(I); 2 = alpha2(I)
    res_index: np.ndarray    # int32: 0 = first helix residue (Gly); <0 N-telo
    uniprot_pos: np.ndarray  # int32: 1-based position in the UniProt entry
    aa: np.ndarray           # '<U1' one-letter code, 'O' = Hyp, '<' '>' termini
    site_type: np.ndarray    # int8 SiteType, current (after modification)
    native_type: np.ndarray  # int8 SiteType before any modification
    s: np.ndarray            # nm, axial position of the side-chain centroid
    phi: np.ndarray          # rad, azimuth of the side-chain centroid
    r: np.ndarray            # nm, radial offset of the side-chain centroid
    s_group: np.ndarray      # nm, axial position of the titratable group
    phi_group: np.ndarray    # rad
    r_group: np.ndarray      # nm
    smeared: np.ndarray      # bool: azimuth uniform (flexible telopeptide)
    hydro: np.ndarray        # raw Kyte-Doolittle index (0 on terminus rows)
    is_residue: np.ndarray   # bool: False on terminus rows
    helix_length_nm: float   # L, axial extent of the helical domain
    d_period_nm: float       # D = 234 residues x axial rise

    def __len__(self) -> int:
        return len(self.site_type)

    @property
    def rise_nm(self) -> float:
        """Axial rise per residue, nm (D / 234)."""
        return self.d_period_nm / D_PERIOD_RESIDUES

    def count(self, site_type: SiteType) -> int:
        return int(np.count_nonzero(self.site_type == int(site_type)))

    def positions(self, which: str = "centroid"):
        """(s, phi, r) of every site. ``which`` is "centroid" or "group"."""
        if which == "centroid":
            return self.s, self.phi, self.r
        if which == "group":
            return self.s_group, self.phi_group, self.r_group
        raise ValueError(f"unknown position model {which!r}")

    def centered_hydrophobicity(self) -> np.ndarray:
        """Kyte-Doolittle index minus its mean over all residues of the
        molecule (helix + telopeptides), the centring the existing tool uses
        (tools/azimuthal_profile.py ``hmean``). Terminus rows are 0."""
        h = np.where(self.is_residue, self.hydro, 0.0)
        return np.where(self.is_residue, h - h[self.is_residue].mean(), 0.0)


# --------------------------------------------------------------- sequences ---
def read_fasta(path) -> str:
    with open(path) as fh:
        return "".join(line.strip() for line in fh if not line.startswith(">"))


def helical_domain(seq: str) -> tuple[int, int]:
    """(0-based start, number of triplets) of the longest uninterrupted
    Gly-X-Y run: the detection rule of tools/build_tropocollagen.py."""
    best = (0, 0)
    for i, aa in enumerate(seq):
        if aa != "G":
            continue
        n = 0
        while i + 3 * n < len(seq) and seq[i + 3 * n] == "G":
            n += 1
        if n > best[1]:
            best = (i, n)
    return best


def chain_segments(data_dir, telopeptides: str = "mature", chains=CHAINS):
    """Per chain type: dict with the helix and telopeptide sequences.

    ``telopeptides``:

    "mature"  intact mature chain ends (acid-soluble collagen);
    "pepsin"  the ends pepsin leaves (PEPSIN_CUTS); the new chain N-termini
              are free alpha-amines;
    "none"    helical domain only (the telopeptide-free limit);
    "tool"    compatibility with tools/azimuthal_profile.py: its 16 / 25
              residue window around the longest Gly-X-Y run. For alpha2 that
              window holds 5 N-propeptide and 10 C-propeptide residues (one
              of them a lysine) and skips the first three telopeptide
              glycines. Kept only so the existing tool can be reproduced.

    In every model but "tool" the helix is the annotated HELIX_RESIDUES
    residues starting at ``ChainSpec.helix_first``; a helix that is not
    Gly-X-Y throughout is rejected.
    """
    if telopeptides not in TELOPEPTIDE_MODELS:
        raise ValueError(f"unknown telopeptide model {telopeptides!r}")
    out = []
    if telopeptides == "tool":
        raw = []
        for spec in chains:
            seq = read_fasta(Path(data_dir) / spec.fasta)
            raw.append((spec, seq) + helical_domain(seq))
        n = min(3 * ntrip for *_, ntrip in raw)
        n -= n % 3
        for spec, seq, start, ntrip in raw:
            end = start + 3 * ntrip
            out.append({"fasta": spec.fasta, "copies": spec.copies,
                        "helix": seq[start:start + n], "helix_start": start,
                        "ntel": seq[max(0, start - TOOL_NTEL):start],
                        "ctel": seq[end:end + TOOL_CTEL], "ctel_first": end})
        return out
    for spec in chains:
        seq = read_fasta(Path(data_dir) / spec.fasta)
        start = spec.helix_first - 1
        helix = seq[start:start + HELIX_RESIDUES]
        if len(helix) != HELIX_RESIDUES or helix[::3] != "G" * (HELIX_RESIDUES // 3):
            raise ValueError(f"{spec.fasta}: residues {spec.helix_first}-"
                             f"{spec.helix_first + HELIX_RESIDUES - 1} are not an "
                             "uninterrupted Gly-X-Y helix; check the ChainSpec")
        ntel = seq[spec.mature_first - 1:start]
        ctel = seq[start + HELIX_RESIDUES:spec.mature_last]
        if telopeptides == "pepsin":
            cut_n, cut_c = PEPSIN_CUTS[spec.fasta]
            ntel = ntel[cut_n:]
            ctel = ctel[:len(ctel) - cut_c]
        elif telopeptides == "none":
            ntel, ctel = "", ""
        out.append({"fasta": spec.fasta, "copies": spec.copies, "helix": helix,
                    "helix_start": start, "ntel": ntel, "ctel": ctel,
                    "ctel_first": start + HELIX_RESIDUES})
    return out


# ---------------------------------------------------------------- geometry ---
def ensure_structure(root=REPO_ROOT) -> Path:
    """Path of ``assets/tropocollagen.pdb``, building it with the existing
    pipeline step if it is missing (``assets/`` is git-ignored)."""
    root = Path(root)
    pdb = root / "assets" / "tropocollagen.pdb"
    meta = root / "assets" / "atoms_meta.json"
    if not (pdb.exists() and meta.exists()):
        proc = subprocess.run([sys.executable, str(root / "tools" / "build_tropocollagen.py")],
                              cwd=root, capture_output=True, text=True)
        if proc.returncode != 0 or not pdb.exists():
            raise RuntimeError("tools/build_tropocollagen.py failed to build the "
                               "full-atom model:\n" + proc.stderr[-2000:])
    return pdb


def read_helix_geometry(pdb_path):
    """Per helix residue: side-chain centroid and titratable-group position.

    Returns dict of arrays keyed (chain 0..2, res_index 0..n-1) flattened in
    PDB order: chain, res_index, resname, centroid xyz (A), group xyz (A),
    plus the minimum atom z (A) that defines the axial origin.
    """
    res = {}
    zmin = np.inf
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            key = ("ABC".index(line[21]), int(line[22:26]) - 1)
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            zmin = min(zmin, xyz[2])
            d = res.setdefault(key, {"rn": line[17:20].strip(), "atoms": {}})
            d["atoms"][name] = xyz
    chain, idx, rn, cen, grp = [], [], [], [], []
    for (ci, ri), d in res.items():
        side = [v for a, v in d["atoms"].items() if a not in _BACKBONE]
        c = np.mean(side, axis=0) if side else np.array(d["atoms"]["CA"])
        names = _GROUP_ATOMS.get(d["rn"])
        g = np.mean([d["atoms"][a] for a in names], axis=0) if names else c
        chain.append(ci); idx.append(ri); rn.append(d["rn"])
        cen.append(c); grp.append(g)
    return {"chain": np.array(chain), "res_index": np.array(idx),
            "resname": np.array(rn), "centroid": np.array(cen),
            "group": np.array(grp), "zmin": float(zmin)}


def _cyl(xyz, zmin):
    """Cartesian A -> (s nm from the helix N-end, phi rad, r nm)."""
    return ((xyz[:, 2] - zmin) / 10.0, np.arctan2(xyz[:, 1], xyz[:, 0]),
            np.hypot(xyz[:, 0], xyz[:, 1]) / 10.0)


def build_native_molecule(root=REPO_ROOT, telopeptides: str = "mature",
                          n_terminus: str = "auto") -> Molecule:
    """Site array of unmodified human type I tropocollagen.

    ``n_terminus``: "free" gives every chain a titratable alpha-amine;
    "blocked" gives none; "auto" (default) blocks a chain whose first residue
    is Gln (pyroglutamate, the state of both intact mature chains) and leaves
    any other N-terminus free, which is what pepsin-cut chains have. A blocked
    N-terminal Gln is typed OTHER: a lactam neither titrates nor deamidates.
    Every chain gets a C-terminal alpha-carboxylate. With
    ``telopeptides="none"`` the termini sit on the helix ends.

    The site array is checked residue by residue against the sequence the
    full-atom model was threaded with; a mismatch (for example a sequence
    file of another species with unchanged CHAINS) raises.
    """
    if n_terminus not in ("auto", "free", "blocked"):
        raise ValueError(f"unknown n_terminus {n_terminus!r}")
    root = Path(root)
    geo = read_helix_geometry(ensure_structure(root))
    meta = json.loads((root / "assets" / "atoms_meta.json").read_text())
    L = float(meta["length_nm"])
    segs = chain_segments(root / "data", telopeptides)
    n = len(segs[0]["helix"])
    if len(geo["chain"]) != 3 * n:
        raise ValueError("full-atom model and sequences differ in helix length")

    cs, cphi, cr = _cyl(geo["centroid"], geo["zmin"])
    gs, gphi, gr = _cyl(geo["group"], geo["zmin"])

    rows = []  # chain, res_index, uniprot, aa, type, s, phi, r, sg, phig, rg, smeared, hydro, is_res

    def add(chain, ri, up, aa, st, s, phi, r, sg, phig, rg, smeared, is_res):
        rows.append((chain, ri, up, aa, int(st), s, phi, r, sg, phig, rg,
                     smeared, KYTE_DOOLITTLE[aa] if is_res else 0.0, is_res))

    chain_seg = [segs[0], segs[0], segs[1]]
    for k in range(len(geo["chain"])):
        ci, ri = int(geo["chain"][k]), int(geo["res_index"][k])
        seg = chain_seg[ci]
        aa = _AA3_TO_1[geo["resname"][k]]
        expect = seg["helix"][ri]
        if (aa if aa != "O" else "P") != expect:
            raise ValueError(f"structure/sequence mismatch at chain {ci} residue {ri}")
        add(ci, ri, seg["helix_start"] + ri + 1, aa, _AA_TO_TYPE.get(aa, SiteType.OTHER),
            cs[k], cphi[k], cr[k], gs[k], gphi[k], gr[k], False, True)

    for ci, seg in enumerate(chain_seg):
        ntel, ctel = seg["ntel"], seg["ctel"]
        blocked = (n_terminus == "blocked"
                   or (n_terminus == "auto" and ntel[:1] == "Q"))
        for j, aa in enumerate(ntel):
            s = -(len(ntel) - j) * TELO_RISE_NM
            st = _AA_TO_TYPE.get(aa, SiteType.OTHER)
            if j == 0 and blocked and aa == "Q":
                st = SiteType.OTHER            # pyroglutamate
            add(ci, j - len(ntel), seg["helix_start"] - len(ntel) + j + 1, aa, st,
                s, 0.0, TELO_RADIUS_NM, s, 0.0, TELO_RADIUS_NM, True, True)
        for j, aa in enumerate(ctel):
            s = L + (j + 1) * TELO_RISE_NM
            add(ci, n + j, seg["ctel_first"] + j + 1, aa,
                _AA_TO_TYPE.get(aa, SiteType.OTHER),
                s, 0.0, TELO_RADIUS_NM, s, 0.0, TELO_RADIUS_NM, True, True)
        # termini: on the outermost telopeptide residue, or on the helix end
        s_n = -len(ntel) * TELO_RISE_NM if ntel else 0.0
        s_c = L + len(ctel) * TELO_RISE_NM if ctel else L
        if not blocked:
            add(ci, -len(ntel), seg["helix_start"] - len(ntel) + 1, "<", SiteType.NTERM,
                s_n, 0.0, TELO_RADIUS_NM, s_n, 0.0, TELO_RADIUS_NM, True, False)
        add(ci, n + len(ctel) - 1, seg["ctel_first"] + len(ctel), ">", SiteType.CTERM,
            s_c, 0.0, TELO_RADIUS_NM, s_c, 0.0, TELO_RADIUS_NM, True, False)

    cols = list(zip(*rows))
    st = np.array(cols[4], dtype=np.int8)
    return Molecule(
        chain=np.array(cols[0], dtype=np.int8),
        res_index=np.array(cols[1], dtype=np.int32),
        uniprot_pos=np.array(cols[2], dtype=np.int32),
        aa=np.array(cols[3], dtype="<U1"),
        site_type=st, native_type=st.copy(),
        s=np.array(cols[5]), phi=np.array(cols[6]), r=np.array(cols[7]),
        s_group=np.array(cols[8]), phi_group=np.array(cols[9]),
        r_group=np.array(cols[10]),
        smeared=np.array(cols[11], dtype=bool),
        hydro=np.array(cols[12]), is_residue=np.array(cols[13], dtype=bool),
        helix_length_nm=L, d_period_nm=float(meta["d_period_nm"]),
    )


@functools.lru_cache(maxsize=None)
def native_molecule(telopeptides: str = "mature", n_terminus: str = "auto") -> Molecule:
    """Cached :func:`build_native_molecule` for the repository's own data."""
    return build_native_molecule(REPO_ROOT, telopeptides, n_terminus)


def make_molecule(s, phi, r, site_type, hydro=None, smeared=None,
                  helix_length_nm=None, d_period_nm=1.0) -> Molecule:
    """Small hand-built molecule (tests, toy models): every row is a residue
    of chain 0, the group position equals the centroid position."""
    s = np.asarray(s, dtype=float)
    n = len(s)
    st = np.asarray(site_type, dtype=np.int8)
    return Molecule(
        chain=np.zeros(n, np.int8), res_index=np.arange(n, dtype=np.int32),
        uniprot_pos=np.arange(1, n + 1, dtype=np.int32),
        aa=np.full(n, "X", dtype="<U1"), site_type=st, native_type=st.copy(),
        s=s, phi=np.asarray(phi, dtype=float), r=np.asarray(r, dtype=float),
        s_group=s.copy(), phi_group=np.asarray(phi, dtype=float).copy(),
        r_group=np.asarray(r, dtype=float).copy(),
        smeared=(np.zeros(n, bool) if smeared is None else np.asarray(smeared, bool)),
        hydro=(np.zeros(n) if hydro is None else np.asarray(hydro, dtype=float)),
        is_residue=np.ones(n, bool),
        helix_length_nm=float(s.max() if helix_length_nm is None else helix_length_nm),
        d_period_nm=float(d_period_nm),
    )
