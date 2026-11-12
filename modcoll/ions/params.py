"""Loader for ``params.toml`` and conversion to numeric tables.

Every parameter in the file is a table with four keys:

    value   the number (or a string for categorical choices)
    units   nm, kT (at 298.15 K), e, mol/L, or "-"
    source  where the number comes from
    status  "literature" | "fitted here" | "provisional"

The loader refuses a parameter that lacks one of them.
"""
from __future__ import annotations

import copy
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as geo
from .energy import KIND_CATION, KIND_OXYANION, MAX_SITES, Elec, Tables, make_elec

PARAMS_PATH = Path(__file__).with_name("params.toml")
STATUSES = ("literature", "fitted here", "provisional")
REQUIRED_KEYS = ("value", "units", "source", "status")

# Avogadro's number times 1e-24 L/nm^3: 1 mol/L = 0.602214 particles per nm^3.
# (N_A = 6.02214076e23 /mol, exact since the 2019 SI.)
PER_NM3_PER_MOLAR = 0.602214076

SPECIES = ("Mg", "Ca", "SO4")

# Which ligand atom types count as a contact partner when contacts and bridges
# are identified. A protonated carboxylic acid is deliberately absent: it is a
# non-ligand here. Sulfate oxygens are partners of the cations (ion pairs).
CATION_PARTNERS = ("O_carboxylate", "O_sulfate")
ANION_PARTNERS = ("N_guanidinium", "N_ammonium", "N_imidazolium")


def _is_leaf(node) -> bool:
    return isinstance(node, dict) and "value" in node


def _walk(node, prefix=""):
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if _is_leaf(child):
            yield path, child
        elif isinstance(child, dict):
            yield from _walk(child, path)


def debye_kappa(ionic_strength: float, bjerrum_length: float) -> float:
    """Inverse Debye length in 1/nm.

    kappa^2 = 8 pi lB N_A I, with I = (1/2) sum_i c_i z_i^2 in mol/L converted
    to number density (Debye & Hueckel 1923; e.g. Israelachvili, Intermolecular
    and Surface Forces, 3rd ed., eq. 14.36). For lB = 0.71 nm this is
    kappa = 3.28 sqrt(I) /nm, i.e. a Debye length of 0.305 / sqrt(I) nm.
    """
    if ionic_strength < 0:
        raise ValueError("ionic strength must be non-negative")
    return math.sqrt(8.0 * math.pi * bjerrum_length * PER_NM3_PER_MOLAR * ionic_strength)


@dataclass
class IonParams:
    """Parsed parameter file plus the choice of calcium coordination number."""

    raw: dict
    ca_coordination: int = 7

    species = SPECIES

    # -- access ------------------------------------------------------------
    def entry(self, dotted: str) -> dict:
        node = self.raw
        for key in dotted.split("."):
            node = node[key]
        if not _is_leaf(node):
            raise KeyError(f"{dotted} is not a parameter")
        return node

    def value(self, dotted: str):
        return self.entry(dotted)["value"]

    def entries(self):
        """All (dotted path, entry) pairs."""
        return list(_walk(self.raw))

    def species_index(self, name: str) -> int:
        return self.species.index(name)

    def with_overrides(self, overrides: dict) -> "IonParams":
        """Copy with some values replaced, {dotted path: value}. Used for
        fitting and for sensitivity scans; the file on disk is not touched."""
        raw = copy.deepcopy(self.raw)
        out = IonParams(raw, self.ca_coordination)
        for dotted, val in overrides.items():
            out.entry(dotted)["value"] = val
        return out

    # -- derived numbers ---------------------------------------------------
    @property
    def bjerrum_length(self) -> float:
        return float(self.value("constants.bjerrum_length"))

    def kappa(self, ionic_strength: float) -> float:
        return debye_kappa(ionic_strength, self.bjerrum_length)

    def elec(self, ionic_strength: float, rcut: float = math.inf) -> Elec:
        return make_elec(self.bjerrum_length, self.kappa(ionic_strength), rcut)

    def polyhedron(self, name: str) -> np.ndarray:
        if name == "Ca":
            key = {6: "geometry_6", 7: "geometry", 8: "geometry_8"}[self.ca_coordination]
        else:
            key = "geometry"
        return geo.POLYHEDRA[self.value(f"species.{name}.{key}")]()

    def free_ion_partition(self, name: str) -> float:
        """Internal partition function of a free ion relative to the fully
        hydrated state: every site is W (weight 1) or L (weight exp(-dG_des)),
        independently, so z_free = (1 + exp(-dG_des))^n. Exactly 1 for sulfate."""
        if name == "SO4":
            return 1.0
        n = len(self.polyhedron(name))
        return (1.0 + math.exp(-float(self.value(f"species.{name}.dg_des")))) ** n

    def tables(self) -> Tables:
        S = len(self.species)
        T = geo.N_LIGAND_TYPES
        v = self.value
        kind = np.zeros(S, dtype=np.int64)
        nsite = np.zeros(S, dtype=np.int64)
        dirs = np.zeros((S, MAX_SITES, 3))
        z = lambda: np.zeros(S)  # noqa: E731
        d_site, q_centre, q_sat, r_sat = z(), z(), z(), z()
        R_centre, R_sat, R_water, dg_des = z(), z(), z(), z()
        w_r, d_out, w_out, d_hb, R_wall, r_short = z(), z(), z(), z(), z(), z()
        w_hb, sin_plane_max = np.ones(S), np.ones(S)
        cos_flat, cos_max = np.ones(S), np.zeros(S)
        eps_in = np.zeros((S, T))
        eps_out = np.zeros((S, T))
        eps_hb = np.zeros((S, T))
        partner = np.zeros((S, T), dtype=np.bool_)
        r_contact = z()
        t_centre = np.full(S, geo.TYPE_INDEX["inert"], dtype=np.int64)
        t_sat = np.full(S, geo.TYPE_INDEX["inert"], dtype=np.int64)

        R_type = np.array([float(v(f"atom_radius.{name}")) for name in geo.LIGAND_TYPES])
        R_max = float(R_type.max())
        contact = self.raw.get("contact", {})

        for i, name in enumerate(self.species):
            U = self.polyhedron(name)
            n = len(U)
            nsite[i] = n
            dirs[i, :n] = U
            pre = f"species.{name}"
            R_wall[i] = v(f"{pre}.wall_radius")
            if name == "SO4":
                kind[i] = KIND_OXYANION
                d_site[i] = v(f"{pre}.d_SO")
                q_sat[i] = v(f"{pre}.oxygen_charge")
                q_centre[i] = float(v(f"{pre}.charge")) - n * q_sat[i]
                r_sat[i] = d_site[i]
                R_centre[i] = R_type[geo.TYPE_INDEX["S_sulfate"]]
                R_sat[i] = R_type[geo.TYPE_INDEX["O_sulfate"]]
                d_hb[i] = v(f"{pre}.hbond_distance")
                w_hb[i] = v(f"{pre}.hbond_halfwidth")
                sin_plane_max[i] = math.sin(math.radians(float(v(f"{pre}.hbond_plane_halfwidth_deg"))))
                t_centre[i] = geo.TYPE_INDEX["S_sulfate"]
                t_sat[i] = geo.TYPE_INDEX["O_sulfate"]
                for tname, block in contact.get(name, {}).items():
                    eps_hb[i, geo.TYPE_INDEX[tname]] = block["eps_hb"]["value"]
                r_contact[i] = v(f"{pre}.outer_contact_distance")
                for tname in ANION_PARTNERS:
                    partner[i, geo.TYPE_INDEX[tname]] = True
                r_short[i] = max(d_site[i] + d_hb[i] + w_hb[i], d_site[i] + R_sat[i] + R_max,
                                 R_centre[i] + R_max) + 0.01
            else:
                kind[i] = KIND_CATION
                d_site[i] = v(f"{pre}.d_MO")
                # what is not on the centre is shared equally among n dummy
                # points (the layout of the multisite models); by default the
                # centre carries everything and the dummies nothing
                q_centre[i] = v(f"{pre}.centre_charge")
                q_sat[i] = (float(v(f"{pre}.charge")) - q_centre[i]) / n
                r_sat[i] = v(f"{pre}.dummy_distance")
                # the ion core meets a ligand oxygen exactly at the bond length
                R_centre[i] = d_site[i] - R_type[geo.TYPE_INDEX["O_carboxylate"]]
                R_water[i] = v(f"{pre}.water_radius")
                dg_des[i] = v(f"{pre}.dg_des")
                w_r[i] = v(f"{pre}.well_radial_halfwidth")
                cos_max[i] = math.cos(math.radians(float(v(f"{pre}.well_angular_halfwidth_deg"))))
                cos_flat[i] = math.cos(math.radians(float(v(f"{pre}.well_angular_flat_deg"))))
                if not cos_flat[i] > cos_max[i]:
                    raise ValueError(f"{name}: well_angular_flat_deg must be below well_angular_halfwidth_deg")
                d_out[i] = v(f"{pre}.outer_shell_distance")
                w_out[i] = v(f"{pre}.outer_shell_halfwidth")
                for tname, block in contact.get(name, {}).items():
                    t = geo.TYPE_INDEX[tname]
                    eps_in[i, t] = block["eps_in"]["value"]
                    eps_out[i, t] = block["eps_out"]["value"]
                # outer-sphere contact ends where the outer shell well ends
                r_contact[i] = d_out[i] + w_out[i]
                for tname in CATION_PARTNERS:
                    partner[i, geo.TYPE_INDEX[tname]] = True
                r_short[i] = max(d_out[i] + w_out[i], d_site[i] + R_water[i] + R_max,
                                 R_centre[i] + R_max, d_site[i] + w_r[i]) + 0.01
        return Tables(
            kind=kind, nsite=nsite, dirs=dirs, d_site=d_site, q_centre=q_centre, q_sat=q_sat,
            r_sat=r_sat, R_centre=R_centre, R_sat=R_sat, R_water=R_water, dg_des=dg_des,
            w_r=w_r, cos_flat=cos_flat, cos_max=cos_max, d_out=d_out, w_out=w_out, eps_in=eps_in, eps_out=eps_out,
            eps_hb=eps_hb, d_hb=d_hb, w_hb=w_hb, sin_plane_max=sin_plane_max, R_type=R_type,
            t_centre=t_centre, t_sat=t_sat, R_wall=R_wall, r_short=r_short,
            partner=partner, r_contact=r_contact,
            eps_wca=float(v("constants.wca_epsilon")),
        )

    def check_site_exclusivity(self) -> None:
        """The inner wells are pairwise additive, so what one site can hold is
        decided by the well's lateral radius d sin(theta_max):

        - below half the O...O distance of a carboxylate (0.111 nm): one oxygen
          per site. Required when ``bidentate_carboxylate`` is "no" (Mg2+).
        - below one oxygen radius (0.140 nm, half the closest approach of
          oxygens of different groups): at most the two oxygens of one
          carboxylate. Required when it is "yes" (Ca2+).

        Raise if a parameter change breaks the rule.
        """
        half_oo = geo.CARBOXYLATE_CO_NM * math.sin(math.radians(geo.CARBOXYLATE_OCO_DEG / 2))
        r_o = float(self.value("atom_radius.O_carboxylate"))
        for name in self.species:
            if name == "SO4":
                continue
            d = float(self.value(f"species.{name}.d_MO"))
            th = math.radians(float(self.value(f"species.{name}.well_angular_halfwidth_deg")))
            allowed = self.value(f"species.{name}.bidentate_carboxylate")
            if allowed not in ("yes", "no"):
                raise ValueError(f"{name}: bidentate_carboxylate must be 'yes' or 'no'")
            limit = r_o if allowed == "yes" else half_oo
            if d * math.sin(th) >= limit:
                raise ValueError(
                    f"{name}: inner well lateral radius {d * math.sin(th):.3f} nm reaches {limit:.3f} nm; "
                    "one site could hold more ligand atoms than the model allows")


def validate_raw(raw: dict) -> None:
    for path, entry in _walk(raw):
        if path.startswith("meta."):
            continue
        missing = [k for k in REQUIRED_KEYS if k not in entry]
        if missing:
            raise ValueError(f"params: {path} lacks {missing}")
        if entry["status"] not in STATUSES:
            raise ValueError(f"params: {path} has status {entry['status']!r}; allowed {STATUSES}")
        if not str(entry["source"]).strip():
            raise ValueError(f"params: {path} has an empty source")


def load_params(path=None, ca_coordination: int = 7, overrides: dict | None = None) -> IonParams:
    """Read the parameter file.

    ca_coordination  7 (pentagonal bipyramid, default), 6 (octahedron) or
                     8 (square antiprism)
    overrides        {dotted path: value} applied after reading
    """
    if ca_coordination not in (6, 7, 8):
        raise ValueError("ca_coordination must be 6, 7 or 8")
    with open(path or PARAMS_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    validate_raw(raw)
    p = IonParams(raw, ca_coordination)
    if overrides:
        p = p.with_overrides(overrides)
    p.check_site_exclusivity()
    return p
