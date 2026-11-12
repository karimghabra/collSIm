"""Explicit multisite divalent ions (Mg2+, Ca2+, SO4 2-) around fixed protein
ligand atoms: energy function, grand-canonical Monte Carlo, calibration of the
contact terms and bridge analysis. See docs/modified_collagen_ions.md.
"""
from .bridges import Bridge, BridgeAnalyzer, BridgeTracker, Contacts, axial_histogram, count_bridges
from .geometry import (LIGAND_TYPES, TYPE_INDEX, LigandSet, ammonium, carboxylate,
                       fraction_deprotonated, guanidinium, imidazolium, place_on_surface,
                       sample_protonation)
from .params import IonParams, debye_kappa, load_params
from .sampler import IonSampler, Snapshot

__all__ = [
    "Association", "Bridge", "BridgeAnalyzer", "BridgeTracker", "Contacts", "IonParams",
    "IonSampler", "LIGAND_TYPES", "LigandSet", "Snapshot", "TYPE_INDEX", "ammonium",
    "association_constants", "axial_histogram", "carboxylate", "count_bridges", "debye_kappa",
    "fraction_deprotonated", "guanidinium", "imidazolium", "load_params", "place_on_surface",
    "sample_protonation",
]


def __getattr__(name):
    # calibrate is imported on first use so that `python -m modcoll.ions.calibrate`
    # does not find the module already loaded
    if name in ("Association", "association_constants"):
        from . import calibrate

        return getattr(calibrate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
