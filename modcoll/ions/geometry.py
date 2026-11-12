"""Coordination polyhedra, rigid-body rotations and protein-side ligand groups.

Units: nm throughout. Charges in elementary charges.

Coordinate convention for anything placed on a collagen molecule: the molecule
axis is +z, the axial coordinate ``s`` is z in nm and the azimuth ``phi`` is
``atan2(y, x)``. This is the convention of ``tools/azimuthal_profile.py``
(``load_residue_surface``: ``s = z / 10`` from the PDB in angstrom,
``phi = arctan2(y, x)`` of the side-chain centroid), so a site table produced
from that logic can be handed to :func:`place_on_surface` unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Ligand atom types. The integer codes index the per-type parameter tables.
# --------------------------------------------------------------------------
LIGAND_TYPES = (
    "O_carboxylate",      # oxygen of a deprotonated carboxylate, -1/2 e each
    "O_carboxylic_acid",  # oxygen of a protonated carboxylic acid, neutral
    "N_guanidinium",      # Arg guanidinium nitrogen, +1/3 e each
    "N_ammonium",         # unmodified Lys ammonium nitrogen, +1 e
    "N_imidazolium",      # protonated His ring nitrogen, +1/2 e each
    "O_sulfate",          # oxygen of a mobile sulfate (set by the sampler)
    "S_sulfate",          # sulfur of a mobile sulfate (set by the sampler)
    "inert",              # excluded volume only
)
TYPE_INDEX = {name: i for i, name in enumerate(LIGAND_TYPES)}
N_LIGAND_TYPES = len(LIGAND_TYPES)

# Group geometry (heavy atoms only).
# Carboxylate: C-O 0.125 nm, O-C-O 125 deg, hence O...O = 2 * 0.125 *
# sin(62.5 deg) = 0.222 nm (Engh & Huber, Acta Cryst. A47 (1991) 392: Asp/Glu
# C-O 1.249 A, O-C-O 123-126 deg).
CARBOXYLATE_CO_NM = 0.125
CARBOXYLATE_OCO_DEG = 125.0
# Guanidinium: planar, C-N 0.133 nm, N-C-N 120 deg, hence N...N = 0.133 *
# sqrt(3) = 0.230 nm (Engh & Huber 1991: Arg CZ-NH 1.326 A, CZ-NE 1.329 A).
GUANIDINIUM_CN_NM = 0.133
# Imidazolium: ND1...NE2 across the ring, 0.22 nm (Engh & Huber 1991 His ring:
# ND1-CE1 1.32 A, CE1-NE2 1.32 A, angle at CE1 about 108-111 deg).
IMIDAZOLIUM_NN_NM = 0.22


# --------------------------------------------------------------------------
# Coordination polyhedra: unit vectors from the ion centre to its sites.
# --------------------------------------------------------------------------
def octahedron() -> np.ndarray:
    """Six sites on +-x, +-y, +-z. Cis angle 90 deg, trans 180 deg."""
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )


def pentagonal_bipyramid() -> np.ndarray:
    """Seven sites: two axial on +-z, five equatorial 72 deg apart."""
    eq = [[np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5), 0.0] for k in range(5)]
    return np.array([[0, 0, 1], [0, 0, -1]] + eq, dtype=float)


def square_antiprism() -> np.ndarray:
    """Eight sites: two squares at polar angle theta and pi - theta, the lower
    one rotated by 45 deg. theta is that of the uniform antiprism, in which all
    sixteen edges are equal.

    Derivation: uniform n-antiprism of edge a has height
    h = a * sqrt(1 - 1 / (4 cos^2(pi / 2n))); for n = 4, h = 0.8409 a. The
    square's circumradius is a / sqrt(2), so tan(theta) = (a / sqrt 2) / (h / 2)
    and theta = 59.26 deg.
    """
    h = np.sqrt(1.0 - 1.0 / (4.0 * np.cos(np.pi / 8.0) ** 2))
    theta = np.arctan2(1.0 / np.sqrt(2.0), h / 2.0)
    out = []
    for k in range(4):
        a = np.pi / 2 * k
        out.append([np.sin(theta) * np.cos(a), np.sin(theta) * np.sin(a), np.cos(theta)])
    for k in range(4):
        a = np.pi / 2 * k + np.pi / 4
        out.append([np.sin(theta) * np.cos(a), np.sin(theta) * np.sin(a), -np.cos(theta)])
    return np.array(out)


def tetrahedron() -> np.ndarray:
    """Four sites on alternating cube corners. Angle acos(-1/3) = 109.47 deg."""
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3.0)


POLYHEDRA = {
    "octahedron": octahedron,
    "pentagonal_bipyramid": pentagonal_bipyramid,
    "square_antiprism": square_antiprism,
    "tetrahedron": tetrahedron,
}


# --------------------------------------------------------------------------
# Quaternions (w, x, y, z), unit norm, active rotation v' = R(q) v.
# --------------------------------------------------------------------------
def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_matrix(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a proper rotation matrix. Uses the
    largest of the four candidate pivots for numerical stability (Shepperd,
    J. Guidance Control 1 (1978) 223)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    cand = np.array([t, R[0, 0], R[1, 1], R[2, 2]])
    i = int(np.argmax(cand))
    if i == 0:
        w = 0.5 * np.sqrt(1.0 + t)
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    elif i == 1:
        x = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)]
    elif i == 2:
        y = 0.5 * np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / (4 * y), (R[0, 1] + R[1, 0]) / (4 * y), y, (R[1, 2] + R[2, 1]) / (4 * y)]
    else:
        z = 0.5 * np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        q = [(R[1, 0] - R[0, 1]) / (4 * z), (R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), z]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product a * b: rotate by b first, then by a."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation: a normalised 4-vector of independent normals is
    uniform on S^3, which double-covers SO(3) uniformly (Shoemake 1992)."""
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def quat_from_two_vectors(a, b) -> np.ndarray:
    """Shortest rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(a @ b)
    if c < -1.0 + 1e-12:  # antiparallel: any perpendicular axis, half turn
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return np.array([0.0, *p])
    q = np.array([1.0 + c, *np.cross(a, b)])
    return q / np.linalg.norm(q)


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """A deterministic unit vector perpendicular to v."""
    trial = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(v, trial)
    return p / np.linalg.norm(p)


# --------------------------------------------------------------------------
# The ligand interface: flat arrays of atoms.
# --------------------------------------------------------------------------
@dataclass
class LigandSet:
    """Fixed protein-side atoms seen by the ions.

    This is the whole interface between a site model and the ion model.

    pos     (N, 3) float, nm
    type    (N,)   int, index into LIGAND_TYPES
    charge  (N,)   float, elementary charges
    mol     (N,)   int, owner molecule id (bridges join two different ids)
    group   (N,)   int, chemical group id; atoms of one carboxylate or one
                   guanidinium share it (used for denticity and bridge maps)
    normal  (N, 3) float, unit normal of the group plane for planar hydrogen
                   bond donors, zero vector when the atom has no plane
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    charge: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mol: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    group: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self):
        self.pos = np.ascontiguousarray(np.asarray(self.pos, dtype=float).reshape(-1, 3))
        n = len(self.pos)
        self.type = np.ascontiguousarray(np.asarray(self.type, dtype=np.int64).reshape(-1))
        self.charge = np.ascontiguousarray(np.asarray(self.charge, dtype=float).reshape(-1))
        self.mol = np.ascontiguousarray(np.asarray(self.mol, dtype=np.int64).reshape(-1))
        if np.size(self.group) == 0 and n:
            self.group = np.arange(n, dtype=np.int64)
        self.group = np.ascontiguousarray(np.asarray(self.group, dtype=np.int64).reshape(-1))
        if np.size(self.normal) == 0 and n:
            self.normal = np.zeros((n, 3))
        self.normal = np.ascontiguousarray(np.asarray(self.normal, dtype=float).reshape(-1, 3))
        for name in ("type", "charge", "mol", "group", "normal"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"LigandSet.{name} has {len(getattr(self, name))} rows, pos has {n}")
        if n and (self.type.min() < 0 or self.type.max() >= N_LIGAND_TYPES):
            raise ValueError("LigandSet.type out of range")

    def __len__(self) -> int:
        return len(self.pos)

    @classmethod
    def concat(cls, parts) -> "LigandSet":
        """Join several sets. Group ids are renumbered so they stay unique."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls()
        groups, offset = [], 0
        for p in parts:
            _, inv = np.unique(p.group, return_inverse=True)
            groups.append(inv + offset)
            offset += int(inv.max()) + 1
        return cls(
            pos=np.concatenate([p.pos for p in parts]),
            type=np.concatenate([p.type for p in parts]),
            charge=np.concatenate([p.charge for p in parts]),
            mol=np.concatenate([p.mol for p in parts]),
            group=np.concatenate(groups),
            normal=np.concatenate([p.normal for p in parts]),
        )

    def translated(self, shift) -> "LigandSet":
        return LigandSet(self.pos + np.asarray(shift, float), self.type, self.charge,
                         self.mol, self.group, self.normal)

    def min_separation(self) -> float:
        """Smallest distance between atoms of different groups (nm).

        The inner-sphere wells are pairwise additive, which is only safe when
        two ligand atoms cannot share one coordination site. Atoms of one group
        are kept apart by the group geometry; this checks the rest.
        """
        n = len(self)
        best = np.inf
        for i in range(n):
            other = self.group != self.group[i]
            if other.any():
                d = np.linalg.norm(self.pos[other] - self.pos[i], axis=1).min()
                best = min(best, float(d))
        return best


def carboxylate(carbon, direction, perp=None, mol=0, protonated=False) -> LigandSet:
    """Two oxygens of a carboxyl group whose carbon sits at ``carbon`` and whose
    C(alpha/beta)->C bond points along ``direction``.

    Deprotonated: -1/2 e on each oxygen (the two are equivalent by resonance).
    Protonated (carboxylic acid): both neutral and of the weak-ligand type.
    ``perp`` fixes the O-C-O plane; by default a deterministic perpendicular.
    """
    c = np.asarray(carbon, float)
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    p = _perpendicular(u) if perp is None else np.asarray(perp, float)
    p = p - (p @ u) * u
    p /= np.linalg.norm(p)
    half = np.radians(CARBOXYLATE_OCO_DEG / 2.0)
    o1 = c + CARBOXYLATE_CO_NM * (np.cos(half) * u + np.sin(half) * p)
    o2 = c + CARBOXYLATE_CO_NM * (np.cos(half) * u - np.sin(half) * p)
    t = TYPE_INDEX["O_carboxylic_acid" if protonated else "O_carboxylate"]
    q = 0.0 if protonated else -0.5
    n = np.cross(u, p)
    return LigandSet(pos=[o1, o2], type=[t, t], charge=[q, q], mol=[mol, mol],
                     group=[0, 0], normal=[n, n])


def guanidinium(carbon, direction, normal=None, mol=0) -> LigandSet:
    """Three coplanar nitrogens around the guanidinium carbon ``carbon``.

    ``direction`` points from the side chain outward. NE sits behind the carbon
    (towards the chain); NH1 and NH2 sit at +-60 deg from ``direction`` and are
    the pair that grips two sulfate oxygens end-on. +1/3 e on each nitrogen
    (the charge is delocalised over the three by resonance).
    """
    c = np.asarray(carbon, float)
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    nrm = _perpendicular(u) if normal is None else np.asarray(normal, float)
    nrm = nrm - (nrm @ u) * u
    nrm /= np.linalg.norm(nrm)
    p = np.cross(nrm, u)
    r = GUANIDINIUM_CN_NM
    ne = c - r * u
    nh1 = c + r * (np.cos(np.pi / 3) * u + np.sin(np.pi / 3) * p)
    nh2 = c + r * (np.cos(np.pi / 3) * u - np.sin(np.pi / 3) * p)
    t = TYPE_INDEX["N_guanidinium"]
    return LigandSet(pos=[ne, nh1, nh2], type=[t] * 3, charge=[1 / 3] * 3, mol=[mol] * 3,
                     group=[0] * 3, normal=[nrm] * 3)


def ammonium(nitrogen, mol=0) -> LigandSet:
    """Lys ammonium nitrogen, +1 e, no preferred plane."""
    return LigandSet(pos=[nitrogen], type=[TYPE_INDEX["N_ammonium"]], charge=[1.0],
                     mol=[mol], group=[0], normal=[[0.0, 0.0, 0.0]])


def imidazolium(centre, direction, normal=None, mol=0) -> LigandSet:
    """The two ring nitrogens of a protonated His, +1/2 e each, straddling
    ``centre`` across ``direction`` in the ring plane."""
    c = np.asarray(centre, float)
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    nrm = _perpendicular(u) if normal is None else np.asarray(normal, float)
    nrm = nrm - (nrm @ u) * u
    nrm /= np.linalg.norm(nrm)
    p = np.cross(nrm, u)
    h = IMIDAZOLIUM_NN_NM / 2
    t = TYPE_INDEX["N_imidazolium"]
    return LigandSet(pos=[c + h * p, c - h * p], type=[t, t], charge=[0.5, 0.5],
                     mol=[mol, mol], group=[0, 0], normal=[nrm, nrm])


GROUP_BUILDERS = {
    "carboxylate": lambda p, u, mol: carboxylate(p, u, mol=mol),
    "carboxylic_acid": lambda p, u, mol: carboxylate(p, u, mol=mol, protonated=True),
    "guanidinium": lambda p, u, mol: guanidinium(p, u, mol=mol),
    "ammonium": lambda p, u, mol: ammonium(p, mol=mol),
    "imidazolium": lambda p, u, mol: imidazolium(p, u, mol=mol),
}


def place_on_surface(s, phi, rho, kinds, mol=0, axis_xy=(0.0, 0.0)) -> LigandSet:
    """Place one ligand group per site on a cylinder around a z-parallel axis.

    s      axial position of each site, nm (z coordinate)
    phi    azimuth of each site, rad, atan2(y, x) about the molecule axis
    rho    radial distance of the group's anchor atom from the axis, nm
    kinds  one key of GROUP_BUILDERS per site
    mol    owner molecule id (scalar or per site)

    Each group points radially outward. The (s, phi) convention is that of
    ``tools/azimuthal_profile.py`` so its site list can be passed straight in.
    """
    s = np.atleast_1d(np.asarray(s, float))
    phi = np.atleast_1d(np.asarray(phi, float))
    rho = np.broadcast_to(np.asarray(rho, float), s.shape)
    mol = np.broadcast_to(np.asarray(mol, dtype=np.int64), s.shape)
    if isinstance(kinds, str):
        kinds = [kinds] * len(s)
    parts = []
    for si, pi_, ri, kind, mi in zip(s, phi, rho, kinds, mol):
        u = np.array([np.cos(pi_), np.sin(pi_), 0.0])
        anchor = np.array([axis_xy[0], axis_xy[1], si]) + ri * u
        parts.append(GROUP_BUILDERS[kind](anchor, u, int(mi)))
    return LigandSet.concat(parts)


# --------------------------------------------------------------------------
# Protonation
# --------------------------------------------------------------------------
def fraction_deprotonated(pH: float, pKa: float) -> float:
    """Henderson-Hasselbalch: [A-] / ([HA] + [A-]) = 1 / (1 + 10^(pKa - pH)).
    Same expression as ``charges_at`` in tools/azimuthal_profile.py."""
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def sample_protonation(n: int, pH: float, pKa: float, rng: np.random.Generator) -> np.ndarray:
    """Independent quenched protonation states for n acids: True = protonated.

    Each site is drawn once and then kept; there is no charge regulation, so an
    approaching cation does not displace the proton.
    """
    return rng.random(n) >= fraction_deprotonated(pH, pKa)
