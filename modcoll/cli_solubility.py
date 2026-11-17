"""Solubility from the pair interaction: native control, modified collagen,
the composition inverse problem with a pair criterion, and the material in
pure water.

    python -m modcoll.cli_solubility [--stages native,variants,modified,inverse,water]
                                     [--pairs 4] [--seed N] [--outdir DIR]

The measure is the aligned-rod second virial coefficient B2 of
``modcoll.solubility``: B2 > 0 means the pair repels on balance (the solution
is stable), B2 < 0 that it attracts (aggregation, precipitation). B2* = B2 /
B2_hc is 1 for hard rods.

native    B2 of unmodified collagen over pH 2-10, ionic strength 1-1000 mM
          and 4, 25, 37 C, for the full energy and for its parts
          (electrostatics alone, hydrophobic term alone, the net charge
          without its pattern with and without the hydrophobic term);
variants  the native control again at a few conditions with the hydrophobic
          strength halved and doubled, the contact distance moved and every
          stagger kept (no in-register window);
modified  the same map, at 25 C plus cold and warm at two pH values, for 90 %
          and complete succinylation, acetylation and deamidation and for the
          representative mixtures of the composition scan, with --pairs
          random pattern pairs each;
inverse   B2 at the bench conditions for compositions on a coarse grid, with
          the isoelectric point of the composition scan: which compositions
          satisfy both the precipitation pH and the pair criteria;
water     the dilution series of the material in deionized water read through
          the charge balance (counter-ions per molecule, where the solution
          sits relative to the isoelectric point) and B2 at each dilution.

Outputs (in --outdir, default results/modcoll/solubility): CSV tables, run.json
and fig_*.png; per-pair tables under raw/ (git-ignored).
"""
from __future__ import annotations

import argparse
import dataclasses
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import cli_common as cc
from . import composition as co
from . import solubility as so
from .cli_stagger import pattern_rng
from .modify import modify
from .sites import SiteType, native_molecule

STAGES = ("native", "variants", "modified", "inverse", "water")

NATIVE_PH = tuple(sorted(set(np.round(np.arange(2.0, 10.01, 0.5), 2).tolist()) | {4.2, 7.4}))
NATIVE_IONIC_MM = (1.0, 3.0, 10.0, 30.0, 50.0, 100.0, 150.0, 300.0, 500.0, 1000.0)
TEMPERATURES_C = (4.0, 25.0, 37.0)

MOD_PH = (3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 7.4)
MOD_IONIC_MM = (1.0, 10.0, 50.0, 150.0, 300.0)
MOD_TEMPERATURE_POINTS = ((4.2, 10.0), (4.2, 150.0), (7.4, 10.0), (7.4, 150.0))

# Compositions of the modified map: (label, f_succ, f_acet, f_deam). The
# mixtures are the representatives of the composition scan
# (results/modcoll/composition/stagger_representatives.csv).
PURE = (("succinylated 100 %", 1.0, 0.0, 0.0), ("succinylated 90 %", 0.9, 0.0, 0.0),
        ("acetylated 100 %", 0.0, 1.0, 0.0), ("acetylated 90 %", 0.0, 0.9, 0.0),
        ("deamidated 100 %", 0.0, 0.0, 1.0), ("deamidated 90 %", 0.0, 0.0, 0.9))
MIXTURES = (("c0 most succinylation", 1.0, 0.0, 0.25),
            ("c1 least succinylation", 0.0, 0.9, 0.95),
            ("c2 most acetylation", 0.0, 1.0, 0.8),
            ("c3 most deamidation", 0.35, 0.4, 1.0),
            ("c5 least acylation", 0.25, 0.0, 1.0),
            ("c6 fill", 0.35, 0.65, 0.0),
            ("c7 everything modified", 1.0, 0.0, 1.0))

# Bench observations as pair criteria: (name, pH, ionic strength mM, expected).
# "soluble" means B2 > 0, "precipitates" B2 < 0. Ionic strengths: deionized
# water with the material's own counter-ions (about 1 mM, see the water
# stage); pH 7 in a dilute buffer; PBS (137 mM NaCl + phosphate, I = 0.16 M);
# 50 mM sodium acetate at pH 4.2 (I = 12 mM) and a saltier acetate.
BENCH = (("deionized water", 3.5, 1.0, "soluble"),
         ("pH 7, dilute buffer", 7.0, 10.0, "soluble"),
         ("PBS", 7.4, 150.0, "soluble"),
         ("sodium acetate pH 4.2", 4.2, 10.0, "precipitates"),
         ("acetate pH 4.2, 50 mM", 4.2, 50.0, "precipitates"))

# pH of 0.5 % w/v material in deionized water and of serial two-fold dilutions
# (bench readings).
WATER_SERIES = ((5.0, 3.27), (2.5, 3.38), (1.25, 3.56), (0.625, 3.79),
                (0.3125, 4.15), (0.15625, 6.61))     # (g/L, pH)

NATIVE_KEY_CONDITIONS = (   # (label, pH, mM, C) for the summary table
    ("dilute acid, low salt", 3.0, 10.0, 25.0),
    ("0.5 M acetic acid-like, pH 2.5, low salt", 2.5, 10.0, 25.0),
    ("dilute acid, 150 mM", 3.0, 150.0, 25.0),
    ("dilute acid, 500 mM", 3.0, 500.0, 25.0),
    ("dilute acid, 1 M", 3.0, 1000.0, 25.0),
    ("neutral, physiological, warm", 7.4, 150.0, 37.0),
    ("neutral, physiological, room", 7.4, 150.0, 25.0),
    ("neutral, physiological, cold", 7.4, 150.0, 4.0),
    ("neutral, low salt", 7.0, 10.0, 25.0),
    ("neutral, 1 M", 7.0, 1000.0, 25.0),
    ("near the isoelectric point", 9.5, 150.0, 25.0),
)

_STATE: dict = {}


# ------------------------------------------------------------- the runner ---
@dataclasses.dataclass
class Job:
    key: tuple                  # identifies the pair and the settings
    virial: str                 # key into the PairVirial table
    mol_a: object
    mol_b: object
    conditions: list
    terms: tuple


def _slab_task(task):
    j, rho = task
    job = _STATE["jobs"][j]
    pv = _STATE["virials"][job.virial]
    return j, rho, pv.slab(job.mol_a, job.mol_b, rho, job.conditions, job.terms)


def run_jobs(jobs, virials, workers: int, progress=None) -> dict:
    """{job key: rows of PairVirial.integrate} for every job, the separation
    slabs of all jobs spread over one worker pool."""
    _STATE.update(jobs=jobs, virials=virials)
    rhos = {j: virials[job.virial].separations(job.conditions) for j, job in enumerate(jobs)}
    tasks = [(j, float(r)) for j in range(len(jobs)) for r in rhos[j]]
    # long-reaching (low-salt, small-rho) slabs first so the pool drains evenly
    tasks.sort(key=lambda t: t[1])
    slabs = {j: {} for j in range(len(jobs))}
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(workers) as pool:
            for k, (j, rho, res) in enumerate(pool.imap_unordered(_slab_task, tasks), 1):
                slabs[j][rho] = res
                if progress and (k % 100 == 0 or k == len(tasks)):
                    progress(k, len(tasks))
    else:
        for k, t in enumerate(tasks, 1):
            j, rho, res = _slab_task(t)
            slabs[j][rho] = res
            if progress and (k % 100 == 0 or k == len(tasks)):
                progress(k, len(tasks))
    out = {}
    for j, job in enumerate(jobs):
        r = rhos[j]
        out[job.key] = virials[job.virial].integrate(
            r, [slabs[j][float(x)] for x in r], job.conditions, job.terms)
    return out


def conditions_grid(ph_values, ionic_mM, temps_C) -> list:
    return [so.Conditions(float(p), 1e-3 * float(i), float(t))
            for t in temps_C for i in ionic_mM for p in ph_values]


def make_pair(native, f_succ, f_acet, f_deam, seed, pair):
    return tuple(modify(native, pattern_rng(seed, pair, k), f_succ=f_succ, f_acet=f_acet,
                        f_deam=f_deam) for k in (0, 1))


def deterministic(f_succ, f_acet, f_deam) -> bool:
    """Every fraction 0 or 1 (with f_succ + f_acet <= 1): every molecule is
    the same, so one pair is the whole ensemble."""
    return all(f in (0.0, 1.0) for f in (f_succ, f_acet, f_deam))


def _mM(row):
    return round(1e3 * row["ionic_strength_M"], 6)


def pair_summary(rows_by_pair: list[list[dict]]) -> list[dict]:
    """Ensemble B2 over random pattern pairs: for a random mixture the
    solution's B2 is the average over pairs, B2 = sum x_i x_j B_ij. Also the
    spread between pairs and the share of pairs that attract."""
    out = []
    for group in zip(*rows_by_pair):
        b = np.array([r["B2_nm3"] for r in group])
        f = np.array([r["F_best_kT"] for r in group])
        first = group[0]
        row = {k: first[k] for k in ("pH", "ionic_strength_M", "temperature_C", "term",
                                     "debye_length_nm", "B2_hc_nm3")}
        n = len(b)
        row.update({
            "pairs": n, "B2_nm3": float(b.mean()),
            "B2_nm3_se": float(b.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "B2_star": float(b.mean() / first["B2_hc_nm3"]),
            "B2_star_min": float(b.min() / first["B2_hc_nm3"]),
            "B2_star_max": float(b.max() / first["B2_hc_nm3"]),
            "share_of_pairs_attractive": float(np.mean(b < 0.0)),
            "F_best_kT_mean": float(f.mean()), "F_best_kT_min": float(f.min()),
            "F_best_kT_max": float(f.max()),
            "soluble": bool(b.mean() > 0.0)})
        best = group[int(np.argmin(f))]
        for k in ("F_best_orientation", "F_best_stagger_nm", "F_best_rho_nm", "E_min_kT",
                  "E_min_el_kT", "E_min_hy_kT", "E_min_orientation", "E_min_stagger_nm"):
            row[f"deepest_pair_{k}"] = best[k]
        out.append(row)
    return out


def flat_rows(rows, **extra) -> list[dict]:
    keep = ("pH", "ionic_strength_M", "temperature_C", "term", "debye_length_nm",
            "B2_nm3", "B2_star", "b2_per_length_nm2", "B2_parallel_nm3",
            "B2_antiparallel_nm3", "F_best_kT", "F_best_orientation", "F_best_stagger_nm",
            "F_best_parallel_kT", "F_best_parallel_stagger_nm", "F_best_antiparallel_kT",
            "F_best_antiparallel_stagger_nm",
            "F_best_rho_nm", "E_min_kT", "E_min_el_kT", "E_min_hy_kT", "E_min_orientation",
            "E_min_stagger_nm", "E_min_rho_nm", "tail_end_parallel", "tail_end_antiparallel",
            "rho_max_nm")
    return [{**extra, **{k: r[k] for k in keep}} for r in rows]


def register_D(stagger_nm, d_nm):
    k = int(round(abs(stagger_nm) / d_nm))
    return k, abs(stagger_nm) - k * d_nm


# ------------------------------------------------------------------ stages ---
def stage_native(native, pv, args, progress):
    conds = conditions_grid(NATIVE_PH, NATIVE_IONIC_MM, TEMPERATURES_C)
    jobs = [Job(("native",), "base", native, native, conds, so.TERMS)]
    rows = run_jobs(jobs, {"base": pv}, args.workers, progress)[("native",)]
    return flat_rows(rows)


def native_key_table(rows, d_nm):
    """The native control at the conditions the established facts refer to,
    with the decomposition into the parts of the energy."""
    by = {(r["pH"], _mM(r), r["temperature_C"], r["term"]): r for r in rows}
    out = []
    for label, pH, mM, t in NATIVE_KEY_CONDITIONS:
        if (pH, mM, t, "full") not in by:
            continue
        full = by[(pH, mM, t, "full")]
        k, off = register_D(full["F_best_stagger_nm"], d_nm)
        row = {"condition": label, "pH": pH, "ionic_mM": mM, "temperature_C": t,
               "net_charge_e": float(co.net_charge_from_counts(_STATE["native_counts"], pH)),
               "B2_star": full["B2_star"], "verdict": "soluble" if full["B2_star"] > 0
               else "attractive",
               "F_best_kT": full["F_best_kT"], "best_orientation": full["F_best_orientation"],
               "best_stagger_nm": full["F_best_stagger_nm"],
               "best_stagger_D": abs(full["F_best_stagger_nm"]) / d_nm,
               "best_at_D_multiple": bool(k >= 1 and abs(off) <= 3.0),
               "best_parallel_kT": full["F_best_parallel_kT"],
               "best_parallel_stagger_D": abs(full["F_best_parallel_stagger_nm"]) / d_nm,
               "E_min_kT": full["E_min_kT"], "E_min_el_kT": full["E_min_el_kT"],
               "E_min_hy_kT": full["E_min_hy_kT"]}
        for term in so.TERMS[1:]:
            row[f"B2_star_{term}"] = by[(pH, mM, t, term)]["B2_star"]
            row[f"F_best_kT_{term}"] = by[(pH, mM, t, term)]["F_best_kT"]
        out.append(row)
    return out


VARIANT_CONDITIONS = ((3.0, 10.0, 25.0), (3.0, 150.0, 25.0), (3.0, 1000.0, 25.0),
                      (4.2, 10.0, 25.0), (7.0, 10.0, 25.0), (7.4, 150.0, 25.0),
                      (7.4, 150.0, 4.0), (7.4, 150.0, 37.0), (7.0, 1000.0, 25.0))
# Explicit parameter changes, each reported next to the base result; none of
# them replaces the base parameters anywhere else.
VARIANTS = (("base", {}),
            ("eps_h x 0.5", {"eps_hydrophobic": 0.025}),
            ("eps_h x 2", {"eps_hydrophobic": 0.1}),
            ("contact 1.55 nm", {"contact_nm": 1.55}),
            ("contact 1.80 nm", {"contact_nm": 1.80}),
            ("contact 2.00 nm", {"contact_nm": 2.00}),
            ("soft core 0.5 nm", {"core_nm": 0.5}),
            ("soft core 0.7 nm", {"core_nm": 0.7}),
            ("contact 1.80 nm, soft core 0.5 nm", {"contact_nm": 1.80, "core_nm": 0.5}),
            ("every stagger kept", {"in_register_exclusion_nm": 0.0}),
            ("no hydrophobic term", {"eps_hydrophobic": 0.0}),
            ("contact term with Kadler 1987 temperature law", {"hydrophobic_law": "kadler1987"}))
VARIANT_MODIFIED = ("succinylated 90 %", "acetylated 100 %", "deamidated 100 %",
                    "c0 most succinylation", "c1 least succinylation", "c3 most deamidation")
NATIVE_FACTS = (   # (pH, mM, C, expected) established for native collagen
    (3.0, 10.0, 25.0, "soluble"), (3.0, 150.0, 25.0, "soluble"),
    (3.0, 1000.0, 25.0, "precipitates"), (7.4, 150.0, 4.0, "soluble"),
    (7.4, 150.0, 37.0, "precipitates"))


def variant_settings(base_settings, name):
    change = dict(VARIANTS)[name]
    return dataclasses.replace(base_settings, **change)


def stage_variants(native, base_settings, args, progress):
    """Native control and the modified bench criteria under every variant:
    which explicit parameter change reproduces which observation."""
    conds = [so.Conditions(p, 1e-3 * i, t) for p, i, t in VARIANT_CONDITIONS]
    bench = [so.Conditions(p, 1e-3 * i) for _, p, i, _ in BENCH]
    comps = {c[0]: c[1:] for c in modified_compositions()}
    names = [n for n, _ in VARIANTS if not args.variants or n in args.variants.split(";")]
    virials, jobs = {}, []
    for name in names:
        virials[name] = so.PairVirial(native, variant_settings(base_settings, name))
        jobs.append(Job((name, "native", 0), name, native, native, conds, ("full",)))
        for label in VARIANT_MODIFIED:
            fs, fa, fd = comps[label]
            n = 1 if deterministic(fs, fa, fd) else min(args.pairs, 2)
            for pair in range(n):
                a, b = make_pair(native, fs, fa, fd, args.seed, pair)
                jobs.append(Job((name, label, pair), name, a, b, bench, ("full",)))
    res = run_jobs(jobs, virials, args.workers, progress)
    rows = []
    for name in names:
        for r in res[(name, "native", 0)]:
            rows.append({"variant": name, "composition": "native", "pair": 0,
                         **flat_rows([r])[0]})
        for label in VARIANT_MODIFIED:
            groups = [res[k] for k in res if k[0] == name and k[1] == label]
            for g_i, g in enumerate(groups):
                for r in g:
                    rows.append({"variant": name, "composition": label, "pair": g_i,
                                 **flat_rows([r])[0]})
    return rows


def variant_verdicts(rows, d_nm: float) -> list[dict]:
    """Which native facts and which modified bench criteria each variant
    reproduces, from the per-pair rows of the variants stage (a pure
    function of those rows, so it can be redone from the saved table)."""
    out = []
    for name in dict.fromkeys(r["variant"] for r in rows):
        sub = [r for r in rows if r["variant"] == name]
        by = {(float(r["pH"]), _mM(r), float(r["temperature_C"])): r
              for r in sub if r["composition"] == "native"}
        v = {"variant": name}
        ok = 0
        for pH, mM, t, expect in NATIVE_FACTS:
            b2 = float(by[(pH, mM, t)]["B2_nm3"])
            hit = (b2 > 0) if expect == "soluble" else (b2 < 0)
            v[f"native pH {pH:g} {mM:g} mM {t:g} C: {expect}"] = bool(hit)
            ok += hit
        warm = by[(7.4, 150.0, 37.0)]
        k, off = register_D(float(warm["F_best_stagger_nm"]), d_nm)
        v["native pH 7.4 150 mM 37 C: D register"] = bool(k >= 1 and abs(off) <= 3.0)
        v["native cold/warm B2 ratio"] = (float(by[(7.4, 150.0, 4.0)]["B2_nm3"])
                                          / float(warm["B2_nm3"]))
        v["native facts reproduced"] = int(ok + v["native pH 7.4 150 mM 37 C: D register"])
        v["native facts tested"] = len(NATIVE_FACTS) + 1
        n_hit = n_all = 0
        for label in VARIANT_MODIFIED:
            for bname, pH, mM, expect in BENCH:
                b2 = [float(r["B2_nm3"]) for r in sub if r["composition"] == label
                      and float(r["pH"]) == pH and _mM(r) == mM]
                mean = float(np.mean(b2))          # B2 of the random mixture
                hit = (mean > 0) if expect == "soluble" else (mean < 0)
                v[f"{label} [{bname}]"] = bool(hit)
                n_hit += hit
                n_all += 1
        v["modified bench criteria met"] = n_hit
        v["modified bench criteria tested"] = n_all
        out.append(v)
    return out


def modified_compositions():
    return [(label, *f) for label, *f in PURE + MIXTURES]


def stage_modified(native, pv, args, progress):
    conds = conditions_grid(MOD_PH, MOD_IONIC_MM, (25.0,))
    conds += [so.Conditions(p, 1e-3 * i, t) for p, i in MOD_TEMPERATURE_POINTS
              for t in (4.0, 37.0)]
    jobs = []
    for label, fs, fa, fd in modified_compositions():
        n = 1 if deterministic(fs, fa, fd) else args.pairs
        for pair in range(n):
            a, b = make_pair(native, fs, fa, fd, args.seed, pair)
            jobs.append(Job((label, pair), "base", a, b, conds, ("full", "el")))
    res = run_jobs(jobs, {"base": pv}, args.workers, progress)
    summary, per_pair = [], []
    for label, fs, fa, fd in modified_compositions():
        groups = [res[k] for k in res if k[0] == label]
        comp = {"composition": label, "f_succ": fs, "f_acet": fa, "f_deam": fd}
        for row in pair_summary(groups):
            summary.append({**comp, **row})
        for pair, g in enumerate(groups):
            per_pair += flat_rows(g, **comp, pair=pair)
    return summary, per_pair


def sample_pI(native, points, molecules, seed):
    """Isoelectric point of the sample at each composition, from the same
    molecules the composition scan draws (same seed and generator)."""
    out = []
    for fs, fa, fd in points:
        counts = co.sample_counts(native, fs, fa, fd, molecules, seed)
        out.append(float(co.isoelectric_point_from_counts(counts.sum(axis=0))))
    return np.array(out)


def stage_inverse(native, base_settings, args, progress):
    pts = [tuple(float(x) for x in p) for p in
           co.composition_grid(args.inverse_step, extra_f_deam=[co.LIMING_F_DEAM])
           if any(np.isclose(p[2], f) for f in args.inverse_f_deam)]
    pI = sample_pI(native, pts, args.molecules, args.seed)
    if args.inverse_band is not None:
        keep = co.in_band(pI, args.target_pH, args.inverse_band)
        pts = [p for p, k in zip(pts, keep) if k]
        pI = pI[keep]
    conds = [so.Conditions(p, 1e-3 * i) for _, p, i, _ in BENCH]
    names = [n for n in args.inverse_variants.split(";") if n]
    virials = {n: so.PairVirial(native, variant_settings(base_settings, n)) for n in names}
    jobs = []
    for name in names:
        for fs, fa, fd in pts:
            n = 1 if deterministic(fs, fa, fd) else args.inverse_pairs
            for pair in range(n):
                a, b = make_pair(native, fs, fa, fd, args.seed, pair)
                jobs.append(Job((name, (fs, fa, fd), pair), name, a, b, conds, ("full",)))
    res = run_jobs(jobs, virials, args.workers, progress)
    rows, summary = [], []
    for name in names:
        r_, s_ = _inverse_tables(name, pts, pI, res, args)
        rows += r_
        summary += s_
    return rows, summary


def _inverse_tables(name, pts, pI, res, args):
    rows = []
    for (fs, fa, fd), pI_k in zip(pts, pI):
        groups = [res[k] for k in res if k[0] == name and k[1] == (fs, fa, fd)]
        summ = pair_summary(groups)
        row = {"variant": name, "f_succ": fs, "f_acet": fa, "f_deam": fd,
               "liming_reference": bool(np.isclose(fd, co.LIMING_F_DEAM)),
               "pI_ensemble": pI_k, "pairs": len(groups)}
        ok_all = True
        for (bname, pH, mM, expect), sm in zip(BENCH, summ):
            row[f"B2_star [{bname}]"] = sm["B2_star"]
            row[f"share_attractive [{bname}]"] = sm["share_of_pairs_attractive"]
            hit = (sm["B2_nm3"] > 0) if expect == "soluble" else (sm["B2_nm3"] < 0)
            row[f"matches [{bname}]"] = bool(hit)
            ok_all &= hit
        row["matches_all_pair_criteria"] = bool(ok_all)
        rows.append(row)
    summary = []
    pI = np.array([r["pI_ensemble"] for r in rows])
    for band in (0.1, 0.2, 0.3, None):
        sel = (np.ones(len(rows), bool) if band is None
               else co.in_band(pI, args.target_pH, band))
        entry = {"variant": name,
                 "pI_band": "any" if band is None else f"{args.target_pH:g} +- {band:g}",
                 "compositions": int(sel.sum())}
        for bname, *_ in BENCH:
            m = np.array([r[f"matches [{bname}]"] for r in rows])
            entry[f"match [{bname}]"] = int((sel & m).sum())
        m = np.array([r["matches_all_pair_criteria"] for r in rows])
        entry["match all pair criteria"] = int((sel & m).sum())
        entry["excluded by the pair criteria"] = int((sel & ~m).sum())
        summary.append(entry)
    return rows, summary


def stage_water(native, pv, args, progress):
    """The dilution series of the material in pure water, through the charge
    balance of ``solubility.solution_pH``, for the native molecule and the
    representative compositions; then B2 at each dilution for a few."""
    mass_native = co.molecular_mass_da(native)
    nat_counts = co.type_counts(native)
    rows, fit_rows = [], []
    comps = [("native", 0.0, 0.0, 0.0)] + modified_compositions()
    for label, fs, fa, fd in comps:
        counts = co.sample_counts(native, fs, fa, fd, args.molecules, args.seed).mean(axis=0)
        mod = co.modification_counts(counts, nat_counts)
        mass = (mass_native + co.MASS_SHIFT_DA["succinyl"] * mod["succinyl"]
                + co.MASS_SHIFT_DA["acetyl"] * mod["acetyl"]
                + co.MASS_SHIFT_DA["deamidation"] * mod["deamidated"])
        pI = float(co.isoelectric_point_from_counts(counts))
        n_each = []
        for g_per_L, pH in WATER_SERIES:
            c = g_per_L / mass
            n_s = so.counter_anions_per_molecule(counts, c, pH)
            n_each.append(n_s)
            rows.append({"composition": label, "f_succ": fs, "f_acet": fa, "f_deam": fd,
                         "pI": pI, "mass_Da": float(mass), "conc_g_per_L": g_per_L,
                         "conc_uM": 1e6 * c, "pH_measured": pH, "pH_minus_pI": pH - pI,
                         "net_charge_at_measured_pH": float(
                             co.net_charge_from_counts(counts, pH)),
                         "counter_anions_per_molecule": n_s,
                         "pH_isoionic": so.solution_pH(counts, c)})
        # one strong-acid stoichiometry for the first five readings
        conc = np.array([gl / mass for gl, _ in WATER_SERIES])
        meas = np.array([p for _, p in WATER_SERIES])
        n_grid = np.arange(-50.0, 300.001, 0.1)
        pred5 = so.solution_pH(counts, conc[None, :5], acid_per_molecule=n_grid[:, None])
        sse5 = ((pred5 - meas[None, :5]) ** 2).sum(axis=1)
        n_fit = float(n_grid[int(np.argmin(sse5))])
        pred = so.solution_pH(counts, conc, acid_per_molecule=n_fit)
        for r, p in zip(rows[-len(WATER_SERIES):], pred):
            r["pH_predicted_fixed_acid"] = float(p)
            r["fitted_anions_per_molecule"] = n_fit
        # alternative: the same stoichiometry plus a strong base in the diluent
        # (constant per volume), fitted to all six readings
        n2 = np.arange(n_fit - 60.0, n_fit + 60.001, 0.5)
        b_uM = np.arange(0.0, 400.001, 2.0)
        pred6 = so.solution_pH(counts, conc[None, None, :], acid_per_molecule=n2[:, None, None],
                               base_M=1e-6 * b_uM[None, :, None])
        sse6 = ((pred6 - meas[None, None, :]) ** 2).sum(axis=2)
        i, j = np.unravel_index(np.argmin(sse6), sse6.shape)
        best = (float(sse6[i, j]), float(n2[i]), float(b_uM[j]))
        basic = float(sum(counts[int(t)] for t in (SiteType.LYS, SiteType.ARG, SiteType.HIS,
                                                    SiteType.NTERM)))
        fit_rows.append({"composition": label, "f_succ": fs, "f_acet": fa, "f_deam": fd,
                         "pI": pI, "basic_groups_per_molecule": basic,
                         "fitted_anions_per_molecule": n_fit,
                         "rms_pH_first_five": float(np.sqrt(sse5.min() / 5)),
                         "pH_predicted_last": pred[-1], "pH_measured_last": WATER_SERIES[-1][1],
                         "anions_from_each_reading": "; ".join(f"{x:.1f}" for x in n_each),
                         "with_diluent_base_anions": best[1], "with_diluent_base_uM": best[2],
                         "with_diluent_base_rms_pH_all_six": float(np.sqrt(best[0] / 6))})
    # B2 at each dilution: ionic strength from the small ions only, protons
    # and the fitted monovalent counter-anions: I = ([H+] + n_s c) / 2
    b2_rows = []
    wanted = [w for w in args.water_compositions.split(";") if w]
    sel = [c for c in comps if c[0] in wanted]
    jobs, meta = [], []
    for label, fs, fa, fd in sel:
        fit = next(f for f in fit_rows if f["composition"] == label)
        counts = co.sample_counts(native, fs, fa, fd, args.molecules, args.seed).mean(axis=0)
        mass = next(r["mass_Da"] for r in rows if r["composition"] == label)
        conds = []
        for g_per_L, pH in WATER_SERIES:
            c = g_per_L / mass
            ionic = max(0.5 * (10.0 ** -pH + max(fit["fitted_anions_per_molecule"], 0.0) * c),
                        1e-4)
            conds.append(so.Conditions(pH, ionic))
        n = 1 if deterministic(fs, fa, fd) else args.pairs
        for pair in range(n):
            a, b = (native, native) if label == "native" else make_pair(
                native, fs, fa, fd, args.seed, pair)
            jobs.append(Job((label, pair), "base", a, b, conds, ("full",)))
        meta.append((label, conds))
    res = run_jobs(jobs, {"base": pv}, args.workers, progress) if jobs else {}
    for label, conds in meta:
        groups = [res[k] for k in res if k[0] == label]
        for (g_per_L, pH), s in zip(WATER_SERIES, pair_summary(groups)):
            b2_rows.append({"composition": label, "conc_g_per_L": g_per_L, "pH": pH,
                            "ionic_strength_mM": 1e3 * s["ionic_strength_M"],
                            "B2_star": s["B2_star"], "B2_star_min": s["B2_star_min"],
                            "B2_star_max": s["B2_star_max"],
                            "share_of_pairs_attractive": s["share_of_pairs_attractive"],
                            "F_best_kT_mean": s["F_best_kT_mean"], "soluble": s["soluble"]})
    return rows, fit_rows, b2_rows


# ------------------------------------------------------------------- main ---
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--pairs", type=int, default=3, help="pattern pairs per composition")
    p.add_argument("--inverse-step", type=float, default=0.25)
    p.add_argument("--inverse-pairs", type=int, default=2)
    p.add_argument("--inverse-band", type=float, default=0.3,
                   help="evaluate only compositions whose isoelectric point is within "
                        "target +- this (the widest band); pass a large value for all")
    p.add_argument("--inverse-f-deam", type=cc.float_list, default=[0.0, 0.18, 0.5, 1.0],
                   help="deamidated fractions of the inverse grid")
    p.add_argument("--variants", default="",
                   help="';'-separated names of VARIANTS to (re)compute in the variants "
                        "stage; earlier rows of the others are kept (default: all)")
    p.add_argument("--inverse-variants",
                   default="base;contact 1.80 nm, soft core 0.5 nm",
                   help="parameter variants (names of VARIANTS, ';'-separated) the "
                        "inverse problem is solved for")
    p.add_argument("--molecules", type=int, default=256,
                   help="molecules per composition for the isoelectric point (as the "
                        "composition scan)")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--target-pH", type=float, default=4.2)
    p.add_argument("--water-compositions",
                   default="native;succinylated 90 %;c0 most succinylation;"
                           "c1 least succinylation;c3 most deamidation")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--outdir", default="results/modcoll/solubility")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--figures-only", action="store_true",
                   help="redraw the figures from the tables already in --outdir")
    p.add_argument("--quick", action="store_true",
                   help="coarse separation grid and few conditions (tests)")
    return p


def _typed(rows):
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    return [{k: conv(v) for k, v in r.items()} for r in rows]


def load_results(outdir: Path) -> dict:
    """The tables a previous run wrote, typed, keyed as ``run`` keeps them in
    memory (for --figures-only)."""
    import csv
    files = {"native_rows": "native_map.csv", "native_key": "native_control.csv",
             "variant_verdicts": "variant_verdicts.csv", "modified_rows": "modified_map.csv",
             "inverse_rows": "inverse_compositions.csv", "water_rows": "water_dilution.csv",
             "water_fits": "water_fit.csv", "water_b2": "water_b2.csv"}
    out = {}
    for key, name in files.items():
        if (outdir / name).exists():
            with open(outdir / name) as fh:
                out[key] = _typed(list(csv.DictReader(fh)))
    return out


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.figures_only:
        from . import solubility_figures as fig
        fig.write_all(Path(args.outdir), load_results(Path(args.outdir)),
                      native_molecule().d_period_nm)
        return 0
    stages = [s for s in args.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    outdir = Path(args.outdir)
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def progress(label):
        return lambda k, n: print(f"  {label}: {k}/{n} slabs ({time.time() - t0:.0f} s)",
                                  flush=True)

    global NATIVE_PH, NATIVE_IONIC_MM, MOD_PH, MOD_IONIC_MM
    settings = so.VirialSettings()
    if args.quick:
        settings = so.VirialSettings(near_offsets_nm=(0.0, 0.1, 0.3, 1.0, 3.0),
                                     far_offsets_nm=(6.0, 12.0, 24.0, 48.0), n_phi=8,
                                     reach_debye=6.0, lag_reach_debye=6.0)
        NATIVE_PH, NATIVE_IONIC_MM = (2.5, 3.0, 7.0, 7.4, 9.5), (10.0, 150.0, 500.0, 1000.0)
        MOD_PH, MOD_IONIC_MM = (3.5, 4.2, 7.0, 7.4), (1.0, 10.0, 150.0)
    native = native_molecule()
    _STATE["native_counts"] = co.type_counts(native)
    pv = so.PairVirial(native, settings)
    d_nm = native.d_period_nm
    # run.json accumulates over runs of single stages into the same directory
    previous = {}
    if (outdir / "run.json").exists():
        import json
        previous = json.loads((outdir / "run.json").read_text())
    run_info = {**previous}
    run_info.update({"stages": sorted(set(previous.get("stages", [])) | set(stages),
                                      key=STAGES.index),
                "seed": args.seed, "pairs": args.pairs,
                "settings": dataclasses.asdict(settings), "B2_hc_nm3": pv.b2_hc,
                "helix_length_nm": pv.length_nm, "D_nm": d_nm,
                "native_pI": float(co.isoelectric_point_from_counts(_STATE["native_counts"])),
                "bench_criteria": [dict(zip(("name", "pH", "ionic_mM", "expected"), b))
                                   for b in BENCH]})
    results = {}
    if "native" in stages:
        rows = stage_native(native, pv, args, progress("native"))
        cc.write_csv(outdir / "native_map.csv", rows)
        key = native_key_table(rows, d_nm)
        cc.write_csv(outdir / "native_control.csv", key)
        results.update(native_rows=rows, native_key=key)
        print(f"native: {len(rows)} rows ({time.time() - t0:.0f} s)", flush=True)
    if "variants" in stages:
        rows = stage_variants(native, settings, args, progress("variants"))
        if args.variants and (outdir / "raw" / "variants_pairs.csv").exists():
            # a selection of variants replaces only its own rows of the saved table
            import csv
            done = {r["variant"] for r in rows}
            with open(outdir / "raw" / "variants_pairs.csv") as fh:
                old = [r for r in _typed(list(csv.DictReader(fh))) if r["variant"] not in done]
            order = [n for n, _ in VARIANTS]
            rows = sorted(old + rows, key=lambda r: order.index(r["variant"]))
        verdicts = variant_verdicts(rows, d_nm)
        cc.write_csv(outdir / "raw" / "variants_pairs.csv", rows)
        cc.write_csv(outdir / "variants.csv", [r for r in rows if r["composition"] == "native"])
        cc.write_csv(outdir / "variant_verdicts.csv", verdicts)
        results.update(variant_rows=rows, variant_verdicts=verdicts)
    if "modified" in stages:
        summary, per_pair = stage_modified(native, pv, args, progress("modified"))
        cc.write_csv(outdir / "modified_map.csv", summary)
        cc.write_csv(outdir / "raw" / "modified_pairs.csv", per_pair)
        results["modified_rows"] = summary
        print(f"modified: {len(summary)} rows ({time.time() - t0:.0f} s)", flush=True)
    if "inverse" in stages:
        rows, summary = stage_inverse(native, settings, args, progress("inverse"))
        cc.write_csv(outdir / "inverse_compositions.csv", rows)
        cc.write_csv(outdir / "inverse_summary.csv", summary)
        results.update(inverse_rows=rows, inverse_summary=summary)
        run_info["inverse"] = summary
        print(f"inverse: {len(rows)} compositions ({time.time() - t0:.0f} s)", flush=True)
    if "water" in stages:
        rows, fits, b2_rows = stage_water(native, pv, args, progress("water"))
        cc.write_csv(outdir / "water_dilution.csv", rows)
        cc.write_csv(outdir / "water_fit.csv", fits)
        cc.write_csv(outdir / "water_b2.csv", b2_rows)
        results.update(water_rows=rows, water_fits=fits, water_b2=b2_rows)
    cc.write_json(outdir / "run.json", run_info)
    if not args.no_figures:
        from . import solubility_figures as fig
        fig.write_all(outdir, results, d_nm)
    print(f"wrote {outdir}/ ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
