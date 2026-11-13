"""Three-way composition scan: succinylation x acetylation x deamidation.

    python -m modcoll.cli_composition [--step 0.025] [--molecules 256] [--seed N]
                                      [--stages map,inverse,sensitivity,assays,stagger]

The composition of the material is not known, so the scan asks the question
the other way round. Over a grid of (f_succ, f_acet, f_deam) it samples
randomly modified molecules (each with its own site-by-site pattern, never a
mean field) and then

map          tabulates the isoelectric point (of the sample and of the single
             molecules), the net charge at pH 4.2 and 7.0 and the charge per
             unit length, each with a standard error;
inverse      finds the compositions whose isoelectric point is the target
             (4.2) within +-0.1, +-0.2 and +-0.3 and that are charged enough at
             pH 7 to count as soluble there;
sensitivity  repeats the inverse problem for every combination of the
             succinyl, Asp and Glu pKa ranges, the two chain-end models and the
             human sequence with or without the four alpha2 histidines that
             bovine and porcine collagen lack;
assays       computes what each bench assay would read for every consistent
             composition and ranks the assays by how well they tell the
             candidates apart;
stagger      runs the two-molecule stagger scan (the S2 machinery, unchanged)
             on a few compositions spanning the consistent set, to see which S2
             conclusions depend on the composition.

Outputs (in --outdir, default results/modcoll/composition):

map          forward_map.csv (the coarse grid and the liming slice),
             pure_chemistries.csv
inverse      region_summary.csv, consistent_region.csv, min_succinylation.csv,
             solubility_threshold.csv
sensitivity  sensitivity.csv, sensitivity_statements.csv,
             sensitivity_min_succinylation.csv
assays       assays.csv, assay_ranking.csv, assay_sets.csv,
             assays_representatives.csv, titration_curves.csv
stagger      stagger_representatives.csv, stagger_summary.csv,
             stagger_conclusions.csv, stagger_crossover.csv

plus run.json and fig_*.png, and under raw/ (large, git-ignored) the tables on
the full fine grid, the sampled type counts, and the stagger minima and
per-pair tables.
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import cli_common as cc
from . import cli_stagger as cs
from . import composition as co
from . import electrostatics as es
from . import stagger as st
from .modify import accessibility_weights, modify
from .sites import SiteType, native_molecule

STAGES = ("map", "inverse", "sensitivity", "assays", "stagger")
CHAIN_END_MODELS = ("mature", "pepsin")
SPECIES = ("human", "bovine_porcine")
BASE_VARIANT = {"succinyl_pKa": 4.5, "asp_pKa": 3.9, "glu_pKa": 4.3,
                "chain_ends": "mature", "species": "human"}
F_DEAM_CAPS = (0.0, 0.1, co.LIMING_F_DEAM, 0.25, 0.5, 0.75, 1.0)
SOLUBLE_THRESHOLDS_E = (0.0, 50.0, 75.0, 100.0, 150.0, 200.0)
TITRATION_PH = np.round(np.arange(2.0, 12.01, 0.1), 2)
S2_REFERENCE = (0.9, 0.0, 0.0)   # 90 % succinylation, the case S2 calls realistic

_STATE: dict = {}    # filled before a worker pool forks


# -------------------------------------------------------------- sampling ---
def _sample_chunk(indices):
    """Type counts of the sampled molecules at each grid point of a chunk:
    their sums, and (if asked for) the per-point statistics."""
    s = _STATE
    sums = np.empty((len(indices), co.N_TYPES), dtype=np.int64)
    stats = []
    for row, i in enumerate(indices):
        f_succ, f_acet, f_deam = (float(x) for x in s["points"][i])
        counts = co.sample_counts(s["native"], f_succ, f_acet, f_deam, s["molecules"],
                                  s["seed"], s["weights"])
        sums[row] = counts.sum(axis=0)
        if s["statistics"]:
            stats.append(co.point_statistics(
                counts, s["pka"], s["pH"], s["length_nm"],
                co.bootstrap_rng(s["seed"], f_succ, f_acet, f_deam), s["n_boot"]))
    return indices, sums, stats


def sample_map(native, points, molecules: int, seed: int, weights=None, workers: int = 1,
               statistics: bool = True, pka=es.DEFAULT_PKA_SET, pH=(4.2, 7.0),
               n_boot: int = 200, progress=None):
    """Sample ``molecules`` molecules at every composition of ``points``.

    Returns (summed type counts [n_points, N_TYPES], list of per-point
    statistics dicts or None). The result does not depend on ``workers``."""
    _STATE.update(native=native, points=np.asarray(points, float), molecules=molecules,
                  seed=seed, weights=weights, statistics=statistics, pka=pka, pH=tuple(pH),
                  length_nm=co.contour_length_nm(native), n_boot=n_boot)
    n = len(points)
    chunks = [np.arange(i, min(i + 64, n)) for i in range(0, n, 64)]
    sums = np.empty((n, co.N_TYPES), dtype=np.int64)
    stats = [None] * n

    def consume(result):
        idx, part, part_stats = result
        sums[idx] = part
        for i, row in zip(idx, part_stats):
            stats[i] = row

    if workers > 1 and len(chunks) > 1:
        ctx = multiprocessing.get_context("fork")     # workers share the site array
        with ctx.Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(_sample_chunk, chunks), 1):
                consume(result)
                if progress and (k % 50 == 0 or k == len(chunks)):
                    progress(k, len(chunks))
    else:
        for chunk in chunks:
            consume(_sample_chunk(chunk))
    return sums, (stats if statistics else None)


# --------------------------------------------------------------- inverse ---
def axis_crossing(f, pI, level: float) -> float:
    """Smallest fraction at which a curve pI(f), sampled at increasing f,
    first comes down to ``level`` (linear interpolation between grid points);
    NaN if it never does."""
    f = np.asarray(f, dtype=float)
    pI = np.asarray(pI, dtype=float)
    if len(f) and pI[0] <= level:
        return float(f[0])
    for k in range(1, len(f)):
        if pI[k - 1] > level >= pI[k]:
            return float(f[k - 1] + (f[k] - f[k - 1]) * (pI[k - 1] - level)
                         / (pI[k - 1] - pI[k]))
    return float("nan")


def pure_axis(points, values, which: int):
    """(f, value) along one pure-chemistry axis of the grid, f increasing."""
    p = np.asarray(points, dtype=float)
    others = [c for c in range(3) if c != which]
    sel = np.flatnonzero((p[:, others[0]] == 0.0) & (p[:, others[1]] == 0.0))
    sel = sel[np.argsort(p[sel, which])]
    return p[sel, which], np.asarray(values)[sel]


def smallest_band(pI, target: float, bands) -> np.ndarray:
    """Half-width of the narrowest band each isoelectric point falls in
    (NaN if it is outside all of them)."""
    out = np.full(len(pI), np.nan)
    for band in sorted(bands, reverse=True):
        out[co.in_band(pI, target, band)] = band
    return out


def solve_inverse(points, pI, q7, target: float, bands, threshold_e: float,
                  caps=F_DEAM_CAPS):
    """Consistent set for each band: region summary rows, the rows of the
    minimum-succinylation table and the boolean masks {band: mask}."""
    summary, min_succ, masks = [], [], {}
    soluble = co.soluble_at(q7, threshold_e)
    for band in bands:
        hit = co.in_band(pI, target, band)
        ok = hit & soluble
        masks[band] = ok
        row = {"band": band, "pI_low": target - band, "pI_high": target + band,
               "soluble_threshold_e": threshold_e,
               "n_grid_points": len(pI), "n_in_band": int(hit.sum()),
               "n_excluded_as_insoluble": int((hit & ~soluble).sum())}
        row.update(co.region_summary(points, pI, ok))
        row["share_of_grid"] = row["n_points"] / len(pI)
        for which, name in enumerate(("succinylation", "acetylation", "deamidation")):
            f, v = pure_axis(points, pI, which)
            row[f"pure_{name}_f_reaching_band"] = axis_crossing(f, v, target + band)
        row["abs_net_charge_pH7_min"] = float(np.abs(q7[ok]).min()) if ok.any() else float("nan")
        row["net_charge_pH7_max"] = float(q7[ok].max()) if ok.any() else float("nan")
        no_succ = ok & (np.asarray(points)[:, 0] == 0.0)
        row["without_succinylation_reached"] = bool(no_succ.any())
        row["without_succinylation_f_deam_min"] = (
            float(np.asarray(points)[no_succ, 2].min()) if no_succ.any() else float("nan"))
        row["without_succinylation_f_acet_min"] = (
            float(np.asarray(points)[no_succ, 1].min()) if no_succ.any() else float("nan"))
        summary.append(row)
        for r in co.minimum_succinylation(points, ok, caps):
            min_succ.append(dict(band=band, **r))
    return summary, min_succ, masks


# ----------------------------------------------------------- sensitivity ---
def variants():
    """Every combination of the scanned parameters, the base case first."""
    out = [dict(BASE_VARIANT)]
    for succ, asp, glu, ends, species in itertools.product(
            co.PKA_RANGES["succinyl"], co.PKA_RANGES["asp"], co.PKA_RANGES["glu"],
            CHAIN_END_MODELS, SPECIES):
        v = {"succinyl_pKa": succ, "asp_pKa": asp, "glu_pKa": glu, "chain_ends": ends,
             "species": species}
        if v != BASE_VARIANT:
            out.append(v)
    return out


def variant_counts(sum_counts, native_counts, molecules: int, species: str):
    """Summed and native type counts for a species. "bovine_porcine" removes
    the four alpha2 histidines; since modification never touches histidine,
    that is the same sample with four fewer His per molecule (the tests
    compare this with sampling the corrected molecule)."""
    sums, nat = np.array(sum_counts), np.array(native_counts)
    if species == "bovine_porcine":
        n_his = len(co.BOVINE_ABSENT_HIS_ALPHA2)
        sums[:, int(SiteType.HIS)] -= n_his * molecules
        sums[:, int(SiteType.OTHER)] += n_his * molecules
        nat[int(SiteType.HIS)] -= n_his
        nat[int(SiteType.OTHER)] += n_his
    elif species != "human":
        raise ValueError(f"unknown species {species!r}")
    return sums, nat


def min_succinylation_curve(points, ok):
    """(caps, least f_succ of any consistent composition with f_deam <= cap),
    one cap per deamidation level of the grid; NaN where nothing qualifies."""
    p = np.asarray(points, dtype=float)
    caps = np.unique(p[:, 2])
    out = np.full(len(caps), np.nan)
    best = np.inf
    for k, cap in enumerate(caps):
        sel = np.asarray(ok, bool) & (p[:, 2] == cap)
        if sel.any():
            best = min(best, float(p[sel, 0].min()))
        if np.isfinite(best):
            out[k] = best
    return caps, out


def sensitivity(points, sums_by_ends, native_by_ends, molecules, target, bands,
                soluble_factor):
    """The inverse problem for every variant.

    Returns (rows, curve rows): one row per variant and band, and for each
    band and deamidation cap the least succinylation in the base case and its
    range over the variants."""
    rows, curves = [], {band: [] for band in bands}
    caps = None
    for v in variants():
        pka = co.pka_variant(succinyl=v["succinyl_pKa"], asp=v["asp_pKa"], glu=v["glu_pKa"])
        sums, nat = variant_counts(sums_by_ends[v["chain_ends"]],
                                   native_by_ends[v["chain_ends"]], molecules, v["species"])
        pI = co.isoelectric_point_from_counts(sums, pka)
        q7 = co.net_charge_from_counts(sums, 7.0, pka) / molecules
        q7_native = float(co.net_charge_from_counts(nat, 7.0, pka))
        summary, min_succ, masks = solve_inverse(points, pI, q7, target, bands,
                                                 soluble_factor * abs(q7_native),
                                                 caps=(0.0, co.LIMING_F_DEAM, 1.0))
        for row in summary:
            by_cap = {r["f_deam_cap"]: r for r in min_succ if r["band"] == row["band"]}
            out = dict(v, is_base=(v == BASE_VARIANT), band=row["band"],
                       native_pI=float(co.isoelectric_point_from_counts(nat, pka)),
                       native_net_charge_pH7=q7_native)
            for key in ("n_points", "share_of_grid", "f_succ_min", "f_acyl_min", "f_deam_max",
                        "pure_succinylation_reaches", "pure_succinylation_f_reaching_band",
                        "pure_succinylation_lowest_pI",
                        "pure_acetylation_reaches", "pure_acetylation_lowest_pI",
                        "pure_deamidation_reaches", "pure_deamidation_lowest_pI",
                        "without_succinylation_reached", "without_succinylation_f_deam_min",
                        "without_succinylation_f_acet_min",
                        "abs_net_charge_pH7_min", "n_excluded_as_insoluble"):
                out[key] = row[key]
            for cap, name in ((0.0, "no_deamidation"), (co.LIMING_F_DEAM, "liming_or_less")):
                r = by_cap[cap]
                out[f"f_succ_min_{name}"] = r.get("f_succ_min", float("nan"))
                out[f"reached_{name}"] = r["reached"]
            out["lowest_pI_anywhere"] = float(np.nanmin(pI))
            rows.append(out)
            caps, curve = min_succinylation_curve(points, masks[row["band"]])
            curves[row["band"]].append(curve)
    curve_rows = []
    for band in bands:
        stack = np.array(curves[band])            # [variant, cap]; variant 0 is the base
        for k, cap in enumerate(caps):
            col = stack[:, k]
            finite = np.isfinite(col)
            curve_rows.append({
                "band": band, "f_deam_cap": float(cap),
                "liming_reference": bool(np.isclose(cap, co.LIMING_F_DEAM)),
                "f_succ_min_base": col[0],
                "f_succ_min_lowest_variant": float(col[finite].min()) if finite.any()
                else float("nan"),
                "f_succ_min_highest_variant": float(col[finite].max()) if finite.any()
                else float("nan"),
                "variants_without_a_solution": int((~finite).sum())})
    return rows, curve_rows


# What is asked of each statement over the variants: (column, kind, wording).
STATEMENTS = (
    ("pure_succinylation_reaches", "bool", "succinylation alone reaches the band"),
    ("pure_acetylation_reaches", "bool", "acetylation alone reaches the band"),
    ("pure_deamidation_reaches", "bool", "deamidation alone reaches the band"),
    ("pure_succinylation_f_reaching_band", "number",
     "succinylated fraction at which succinylation alone enters the band"),
    ("without_succinylation_reached", "bool", "a composition without any succinylation exists"),
    ("without_succinylation_f_deam_min", "number",
     "least deamidation of a composition without succinylation"),
    ("without_succinylation_f_acet_min", "number",
     "least acetylation of a composition without succinylation"),
    ("f_succ_min", "number", "least succinylation of any consistent composition"),
    ("reached_liming_or_less", "bool",
     "a composition with deamidation at or below the liming level exists"),
    ("f_succ_min_liming_or_less", "number",
     "least succinylation with deamidation at or below the liming level"),
    ("reached_no_deamidation", "bool", "a composition without deamidation exists"),
    ("f_succ_min_no_deamidation", "number", "least succinylation without deamidation"),
    ("f_acyl_min", "number", "least total acylation (succinyl + acetyl) of any composition"),
    ("abs_net_charge_pH7_min", "number",
     "smallest net charge magnitude at pH 7 of any composition (e)"),
    ("n_excluded_as_insoluble", "number",
     "compositions in the band excluded by the pH 7 solubility proxy"),
    ("share_of_grid", "number", "share of all grid compositions that are consistent"),
)


def statements(sens_rows, bands):
    """For each band and statement: the base-case value and what the variants
    do to it. A yes/no statement "survives" if all variants agree with the
    base case; for a number the range over the variants is given (a variant
    in which the quantity does not exist, because nothing qualifies, is
    counted separately)."""
    out = []
    for band in bands:
        rows = [r for r in sens_rows if r["band"] == band]
        base = next(r for r in rows if r["is_base"])
        for key, kind, wording in STATEMENTS:
            row = {"band": band, "statement": wording, "quantity": key,
                   "n_variants": len(rows)}
            if kind == "bool":
                values = [bool(r[key]) for r in rows]
                row.update({"base": bool(base[key]), "true_in_variants": int(sum(values)),
                            "same_in_all_variants": len(set(values)) == 1})
            else:
                values = np.array([r[key] for r in rows], dtype=float)
                finite = np.isfinite(values)
                row.update({"base": base[key],
                            "min_over_variants": float(values[finite].min()) if finite.any()
                            else float("nan"),
                            "max_over_variants": float(values[finite].max()) if finite.any()
                            else float("nan"),
                            "variants_undefined": int((~finite).sum())})
            out.append(row)
    return out


def one_at_a_time(sens_rows, band):
    """Rows of the sensitivity table in which exactly one parameter differs
    from the base case (and the base case itself), for one band."""
    rows = []
    for r in sens_rows:
        if r["band"] != band:
            continue
        changed = [k for k in BASE_VARIANT if r[k] != BASE_VARIANT[k]]
        if len(changed) <= 1:
            rows.append(dict(changed=(changed[0] if changed else "base"), **r))
    return rows


# ---------------------------------------------------------------- stagger ---
def crossover_salt_mM(ionic_mM, share, level: float = 0.5):
    """Ionic strength at which the share of pattern pairs whose deepest
    minimum sits at a D multiple rises through ``level``, interpolated
    linearly in log10 of the ionic strength between the scanned values.

    Returns (value in mmol/L, status). Status "bracketed": the last upward
    crossing lies between two scanned values. "below_range": the share is at
    or above the level at every scanned ionic strength (value = the lowest
    one, an upper bound). "above_range": it is below the level at every one
    (value NaN)."""
    x = np.asarray(ionic_mM, dtype=float)
    y = np.asarray(share, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.all(y >= level):
        return float(x[0]), "below_range"
    if np.all(y < level):
        return float("nan"), "above_range"
    if y[-1] < level:
        return float("nan"), "not_monotonic"
    k = max(i for i in range(1, len(x)) if y[i - 1] < level <= y[i])
    t = (level - y[k - 1]) / (y[k] - y[k - 1])
    return float(10.0 ** (np.log10(x[k - 1]) + t * (np.log10(x[k]) - np.log10(x[k - 1])))), \
        "bracketed"


def _stagger_eval(task):
    """Profiles and per-pair descriptors of one pattern pair at one
    composition, for every pH and ionic strength (cli_stagger._evaluate with
    the three fractions given explicitly)."""
    comp, pair = task
    s = _STATE
    model = s["model"]
    f_succ, f_acet, f_deam = s["compositions"][comp]
    mol_a, mol_b = (modify(s["native"], cs.pattern_rng(s["seed"], pair, k), f_succ=f_succ,
                           f_acet=f_acet, f_deam=f_deam, weights=s["weights"])
                    for k in (0, 1))
    prof = model.profiles(mol_a, mol_b, s["ph"], s["charge_model"])
    d_nm = s["native"].d_period_nm
    stacked, metrics = {}, []
    for o in model.orientations:
        stacked[o] = np.stack([np.stack([prof[(pH, ionic, o)] for ionic in s["ionic"]])
                               for pH in s["ph"]]).astype(np.float32)
    for ip, pH in enumerate(s["ph"]):
        env = es.Environment(pH=pH)
        qa = es.net_charge(mol_a, env, s["charge_model"])
        qb = es.net_charge(mol_b, env, s["charge_model"])
        for ii, ionic in enumerate(s["ionic"]):
            row = {"f": float(comp), "pair": pair, "pH": pH, "ionic_mM": 1e3 * ionic,
                   "net_charge_a": qa, "net_charge_b": qb}
            if "parallel" in stacked:
                row.update(cs.pair_metrics(model.lags["parallel"],
                                           stacked["parallel"][ip, ii].astype(float), d_nm))
            if "antiparallel" in stacked:
                u = stacked["antiparallel"][ip, ii, 0]
                j = int(np.argmin(u))
                row.update({"anti_deepest_nm": model.lags["antiparallel"][j],
                            "anti_deepest_kT": float(u[j])})
            metrics.append(row)
    return comp, pair, stacked, metrics


def stagger_scan(native, compositions, ph, ionic_M, pairs: int, seed: int,
                 pka_set: str = es.DEFAULT_PKA_SET, orientations=("parallel", "antiparallel"),
                 weights=None, workers: int = 1, settings=None, progress=None):
    """The S2 two-molecule stagger scan at explicit compositions.

    Pattern pair k uses the generators of ``cli_stagger.pattern_rng``, so a
    composition that is a pure chemistry gives exactly what ``cli_stagger``
    gives for it. A composition without randomness (every fraction 0 or 1)
    is evaluated once. Returns (summary rows, minima rows, per-pair rows,
    native gate), the first two as ``cli_stagger.summarise`` defines them
    with the composition in place of the modified fraction."""
    settings = settings or st.PairSettings()
    d_nm = native.d_period_nm
    gate_model = st.PairModel(native, [0.15], settings, ("parallel",))
    gate_prof = gate_model.profiles(native, native, [7.4], es.charge_model_for(
        "repository"))[(7.4, 0.15, "parallel")]
    gate = st.native_gate(gate_model.lags["parallel"], gate_prof[0], gate_prof[2], d_nm)
    gate["pka_set"] = "repository"
    if not gate["passed"]:
        return [], [], [], gate

    compositions = [tuple(float(x) for x in c) for c in compositions]
    model = st.PairModel(native, list(ionic_M), settings, tuple(orientations))
    _STATE.update(model=model, native=native, compositions=compositions, seed=seed,
                  weights=weights, ph=list(ph), ionic=list(ionic_M),
                  charge_model=es.charge_model_for(pka_set))
    tasks = []
    for c, comp in enumerate(compositions):
        random = any(0.0 < f < 1.0 for f in comp)
        tasks += [(c, pair) for pair in range(pairs if random else 1)]
    shape = (len(compositions), len(ph), len(ionic_M), len(st.REDUCTIONS))
    acc = {o: np.zeros(shape + (len(model.lags[o]),)) for o in orientations}
    acc2 = {o: np.zeros(shape + (len(model.lags[o]),)) for o in orientations}
    count = np.zeros(len(compositions), int)
    metrics = []

    def consume(result):
        c, _, stacked, rows = result
        for o in orientations:
            x = stacked[o].astype(float)
            acc[o][c] += x
            acc2[o][c] += x * x
        count[c] += 1
        metrics.extend(rows)

    if workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("fork")     # workers share the kernel tables
        with ctx.Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(_stagger_eval, tasks), 1):
                consume(result)
                if progress and (k % 32 == 0 or k == len(tasks)):
                    progress(k, len(tasks))
    else:
        for task in tasks:
            consume(_stagger_eval(task))
    metrics.sort(key=lambda m: (m["f"], m["pair"], m["pH"], m["ionic_mM"]))

    mean, sd = {}, {}
    for o in orientations:
        n = count[:, None, None, None, None]
        mean[o] = acc[o] / n
        var = np.clip(acc2[o] / n - mean[o] ** 2, 0.0, None) * (n / np.maximum(n - 1, 1))
        sd[o] = np.where(n > 1, np.sqrt(var), 0.0)

    ns = SimpleNamespace(chemistry="composition", pka_set=pka_set, f_deam=float("nan"),
                         ph=list(ph), ionic=list(ionic_M))
    ids = [float(c) for c in range(len(compositions))]
    summary, minima = cs.summarise(ns, model, d_nm, ids, mean, sd, metrics)

    def relabel(row):
        comp = compositions[int(row["f"])]
        rest = {k: v for k, v in row.items() if k not in ("f", "chemistry", "f_deam")}
        return dict(composition=int(row["f"]), f_succ=comp[0], f_acet=comp[1],
                    f_deam=comp[2], **rest)

    return ([relabel(r) for r in summary], [relabel(r) for r in minima],
            [relabel(r) for r in metrics], gate)


# The S2 conclusions that are checked against the composition, with the
# criterion each is held to. "Holds" means the criterion is met at every
# representative composition.
D_SHARE_HOLDS = 0.9          # "the D register survives": at least this share of pairs
D_SHARE_LOST = 0.1           # "the D register is lost": at most this share of pairs
PERIOD_TOLERANCE_NM = 2.0    # "repeats with the D period"
DEPTH_SPREAD_KT = 5.0        # "the same depth": range over compositions
SPREAD_HIGH_SALT_KT = 2.0    # S2: pair-to-pair sd up to 1.5 kT at 150 mM
SPREAD_LOW_SALT_KT = 6.0     # S2: 3.7-5.7 kT at 10 mM
CROSSOVER_FACTOR = 2.0       # "the same crossover": max / min over compositions


def stagger_conclusions(summary, d_nm: float, target_pH: float = 4.2, own_pI=None):
    """One row per S2 conclusion: the number it is read from at every
    representative composition (columns c0, c1, ...), its range, and whether
    the criterion holds at all of them. ``own_pI`` maps a composition to its
    isoelectric point; given it, the low-salt and crossover conclusions are
    also evaluated at the scanned pH nearest each composition's own
    isoelectric point."""
    par = [r for r in summary if r["orientation"] == "parallel"]
    anti = [r for r in summary if r["orientation"] == "antiparallel"]
    comps = sorted({r["composition"] for r in par})
    salts = sorted({r["ionic_mM"] for r in par})
    phs = sorted({r["pH"] for r in par})
    hi, lo = salts[-1], salts[0]

    def rows(pool=par, **where):
        return [r for r in pool if all(np.isclose(r[k], v) for k, v in where.items())]

    def one(c, salt, pH, key, pool=par):
        return rows(pool, composition=c, ionic_mM=salt, pH=pH)[0][key]

    def crossover(c, pH):
        sel = sorted(rows(composition=c, pH=pH), key=lambda r: r["ionic_mM"])
        return crossover_salt_mM([r["ionic_mM"] for r in sel],
                                 [r["pairs_deepest_at_multiple_mean"] for r in sel])

    out = []

    def add(key, wording, unit, criterion, values, fails, note=None):
        v = np.array([values[c] for c in comps], dtype=float)
        finite = np.isfinite(v)
        row = {"conclusion": key, "statement": wording, "unit": unit, "criterion": criterion,
               "holds_at_all_compositions": not fails,
               "min_over_compositions": float(v[finite].min()) if finite.any() else float("nan"),
               "max_over_compositions": float(v[finite].max()) if finite.any() else float("nan"),
               "compositions_where_it_fails": " ".join(str(c) for c in fails),
               "note": note or ""}
        row.update({f"c{c}": values[c] for c in comps})
        out.append(row)

    def each(key, wording, unit, criterion, fn, ok):
        values = {c: fn(c) for c in comps}
        add(key, wording, unit, criterion, values, [c for c in comps if not ok(values[c])])

    def spread(key, wording, unit, criterion, fn, width):
        values = {c: fn(c) for c in comps}
        v = np.array(list(values.values()), dtype=float)
        fails = ([] if np.ptp(v) <= width else
                 [c for c in comps if abs(values[c] - np.median(v)) > 0.5 * width])
        add(key, wording, unit, criterion, values, fails)

    def crossovers(key, wording, ph_of):
        values, status = {}, {}
        for c in comps:
            values[c], status[c] = crossover(c, ph_of(c))
        finite = [v for v in values.values() if np.isfinite(v)]
        fails = [c for c in comps if status[c] != "bracketed"]
        if not fails and finite and max(finite) / min(finite) > CROSSOVER_FACTOR:
            mid = float(np.exp(np.mean(np.log(finite))))
            fails = [c for c in comps
                     if max(values[c] / mid, mid / values[c]) > np.sqrt(CROSSOVER_FACTOR)]
        add(key, wording, "mM",
            f"bracketed by the scan at every composition and within a factor "
            f"{CROSSOVER_FACTOR:g} across them", values, fails,
            note=" ".join(f"c{c}:{status[c]}" for c in comps if status[c] != "bracketed"))

    each("register_at_high_salt",
         f"at {hi:g} mM a pair's deepest minimum is at a D multiple at every pH scanned "
         "(least share of pairs over the pH values)", "share of pairs",
         f">= {D_SHARE_HOLDS:g}",
         lambda c: min(r["pairs_deepest_at_multiple_mean"]
                       for r in rows(composition=c, ionic_mM=hi)),
         lambda x: x >= D_SHARE_HOLDS)
    each("register_is_1D",
         f"at {hi:g} mM and pH {target_pH:g} the deepest minimum of the mean profile is at 1 D",
         "D", "within 0.05 of 1", lambda c: one(c, hi, target_pH, "deepest_D"),
         lambda x: abs(x - 1.0) <= 0.05)
    each("period_at_high_salt",
         f"at {hi:g} mM and pH {target_pH:g} the profile repeats with the D period", "nm",
         f"within {PERIOD_TOLERANCE_NM:g} nm of D = {d_nm:.1f} nm",
         lambda c: one(c, hi, target_pH, "dominant_period_nm"),
         lambda x: abs(x - d_nm) <= PERIOD_TOLERANCE_NM)
    spread("depth_at_high_salt",
           f"depth of a pair's deepest minimum at {hi:g} mM and pH {target_pH:g}", "kT",
           f"range over compositions <= {DEPTH_SPREAD_KT:g} kT",
           lambda c: one(c, hi, target_pH, "deepest_kT_pairs_mean"), DEPTH_SPREAD_KT)
    each("electrostatics_does_not_select_D",
         f"D-comb z-score of the electrostatic term alone at {hi:g} mM and pH {target_pH:g}",
         "z", "< 1", lambda c: one(c, hi, target_pH, "el_comb_z_mean_profile"),
         lambda x: x < 1.0)
    each("register_lost_at_low_salt",
         f"at {lo:g} mM and pH {target_pH:g} a pair's deepest minimum is not at a D multiple",
         "share of pairs", f"<= {D_SHARE_LOST:g}",
         lambda c: one(c, lo, target_pH, "pairs_deepest_at_multiple_mean"),
         lambda x: x <= D_SHARE_LOST)
    each("register_lost_at_low_salt_all_pH",
         f"the same at every pH scanned (largest share of pairs over the pH values, {lo:g} mM)",
         "share of pairs", f"<= {D_SHARE_LOST:g}",
         lambda c: max(r["pairs_deepest_at_multiple_mean"]
                       for r in rows(composition=c, ionic_mM=lo)),
         lambda x: x <= D_SHARE_LOST)
    spread("low_salt_register",
           f"stagger of the deepest minimum of the mean profile at {lo:g} mM and pH "
           f"{target_pH:g}", "D", "the same register (within 0.05 D) at every composition",
           lambda c: one(c, lo, target_pH, "deepest_D"), 0.05)
    for pH in phs:
        crossovers(f"crossover_salt_pH{pH:g}",
                   f"ionic strength at which half the pairs prefer a D multiple, pH {pH:g}",
                   lambda c, pH=pH: pH)
    if own_pI is not None:
        def nearest(c):
            return min(phs, key=lambda pH: abs(pH - own_pI[c]))

        each("register_lost_at_low_salt_own_pI",
             f"at {lo:g} mM and the scanned pH nearest the composition's own isoelectric "
             "point, a pair's deepest minimum is not at a D multiple", "share of pairs",
             f"<= {D_SHARE_LOST:g}",
             lambda c: one(c, lo, nearest(c), "pairs_deepest_at_multiple_mean"),
             lambda x: x <= D_SHARE_LOST)
        crossovers("crossover_salt_own_pI",
                   "ionic strength at which half the pairs prefer a D multiple, at the "
                   "scanned pH nearest the composition's own isoelectric point", nearest)
    for salt, limit in ((hi, SPREAD_HIGH_SALT_KT), (lo, SPREAD_LOW_SALT_KT)):
        each(f"pair_spread_{salt:g}mM",
             f"pair-to-pair sd of the deepest minimum at {salt:g} mM and pH {target_pH:g}",
             "kT", f"<= {limit:g}", lambda c, salt=salt: one(c, salt, target_pH,
                                                             "deepest_kT_pairs_sd"),
             lambda x, limit=limit: x <= limit)
    if anti:
        each("no_polarity_selection",
             f"antiparallel minus parallel deepest minimum at {hi:g} mM and pH {target_pH:g}",
             "kT", "< 2 (antiparallel pairs are not disfavoured)",
             lambda c: one(c, hi, target_pH, "anti_minus_parallel_kT_mean", anti),
             lambda x: x < 2.0)
    return out


def crossover_table(summary):
    """Crossover ionic strength for every representative composition and pH."""
    par = [r for r in summary if r["orientation"] == "parallel"]
    out = []
    for c in sorted({r["composition"] for r in par}):
        for pH in sorted({r["pH"] for r in par}):
            sel = sorted((r for r in par if r["composition"] == c and np.isclose(r["pH"], pH)),
                         key=lambda r: r["ionic_mM"])
            value, status = crossover_salt_mM([r["ionic_mM"] for r in sel],
                                              [r["pairs_deepest_at_multiple_mean"] for r in sel])
            out.append({"composition": c, "f_succ": sel[0]["f_succ"], "f_acet": sel[0]["f_acet"],
                        "f_deam": sel[0]["f_deam"], "pH": pH,
                        "net_charge_mean": sel[0]["net_charge_mean"],
                        "crossover_mM": value, "status": status})
    return out


# ------------------------------------------------------------------- main ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--step", type=float, default=0.025,
                   help="grid spacing of all three fractions")
    p.add_argument("--coarse-step", type=float, default=0.1,
                   help="spacing of the committed forward map and of the figures' "
                        "deamidation slices (a multiple of --step)")
    p.add_argument("--assay-step", type=float, default=0.05,
                   help="spacing of the candidate compositions the assays are ranked on")
    p.add_argument("--molecules", type=int, default=256, help="molecules per composition")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="bootstrap resamples for the standard error of the sample pI")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--target-pH", type=float, default=4.2,
                   help="the isoelectric point to match")
    p.add_argument("--bands", type=cc.float_list, default=[0.1, 0.2, 0.3],
                   help="half-widths of the tolerance bands on the isoelectric point")
    p.add_argument("--soluble-factor", type=float, default=2.0,
                   help="soluble at pH 7 = net charge magnitude at least this many times "
                        "that of the unmodified molecule at pH 7")
    p.add_argument("--weights", choices=("uniform", "crosslink_sites", "ion_pair"),
                   default="uniform", help="site reactivity model")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--representatives", type=int, default=8,
                   help="compositions for the stagger scan (plus the S2 reference)")
    p.add_argument("--stagger-ph", type=cc.float_list, default=[3.8, 4.0, 4.2, 4.4, 4.6])
    p.add_argument("--stagger-ionic-mM", type=cc.float_list,
                   default=[10.0, 25.0, 50.0, 100.0, 150.0])
    p.add_argument("--pairs", type=int, default=64, help="pattern pairs per composition")
    p.add_argument("--orientations", default="parallel,antiparallel")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--outdir", default="results/modcoll/composition")
    p.add_argument("--no-figures", action="store_true")
    return p


def _composition_row(point):
    return {"f_succ": float(point[0]), "f_acet": float(point[1]), "f_deam": float(point[2])}


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stages = [s for s in args.stages.split(",") if s]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {STAGES}")
    outdir = Path(args.outdir)
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bands = sorted(args.bands)
    target = args.target_pH
    base_pka = co.pka_variant()

    def progress(label):
        return lambda k, n: print(f"  {label}: {k}/{n}  ({time.time() - t0:.0f} s)", flush=True)

    # ---- map: sample every composition, for both chain-end models -------------
    points = co.composition_grid(args.step, extra_f_deam=[co.LIMING_F_DEAM])
    natives = {ends: native_molecule(ends) for ends in CHAIN_END_MODELS}
    native_counts = {ends: co.type_counts(m) for ends, m in natives.items()}
    need_pepsin = "sensitivity" in stages
    sums, stats = {}, None
    for ends in CHAIN_END_MODELS:
        if ends != "mature" and not need_pepsin:
            continue
        weights = accessibility_weights(natives[ends], args.weights)
        sums[ends], s = sample_map(
            natives[ends], points, args.molecules, args.seed, weights, args.workers,
            statistics=(ends == "mature"), pka=base_pka, pH=(target, 7.0),
            n_boot=args.bootstrap, progress=progress(f"sampling, {ends} chain ends"))
        if ends == "mature":
            stats = s
    np.savez_compressed(raw / "sampled_counts.npz", points=points,
                        site_types=np.array([t.name for t in SiteType]),
                        **{f"sum_counts_{k}": v for k, v in sums.items()},
                        molecules=args.molecules, seed=args.seed)

    native = natives["mature"]
    nat = native_counts["mature"]
    length = co.contour_length_nm(native)
    mass = co.molecular_mass_da(native)
    mean_counts = sums["mature"] / args.molecules
    mod = co.modification_counts(mean_counts, nat)
    pI = np.array([s["pI_ensemble"] for s in stats])
    q7 = np.array([s["net_charge_pH7_mean"] for s in stats])
    tag = f"pH{target:g}"
    liming = np.isclose(points[:, 2], co.LIMING_F_DEAM)
    q7_native = float(co.net_charge_from_counts(nat, 7.0, base_pka))
    threshold = args.soluble_factor * abs(q7_native)

    map_rows = []
    for i, s in enumerate(stats):
        row = _composition_row(points[i])
        row["liming_reference"] = bool(liming[i])
        row.update(s)
        row.update({"mean_succinylated": mod["succinyl"][i], "mean_acetylated": mod["acetyl"][i],
                    "mean_deamidated": mod["deamidated"][i],
                    "mean_free_amines": mod["free_amines"][i]})
        map_rows.append(row)
    coarse = co.on_grid(points[:, :2], args.coarse_step) & (
        co.on_grid(points[:, 2:], args.coarse_step) | liming)
    cc.write_csv(outdir / "forward_map.csv", [r for r, keep in zip(map_rows, coarse) if keep])
    cc.write_csv(raw / "forward_map_fine.csv", map_rows)

    pure_rows = []
    order = np.arange(len(points))
    for which, name in enumerate(("succinylation", "acetylation", "deamidation")):
        for f, i in zip(*pure_axis(points, order, which)):
            pure_rows.append({"chemistry": name, "f": float(f),
                              **{k: map_rows[i][k] for k in (
                                  "pI_ensemble", "pI_ensemble_se", "pI_molecule_sd",
                                  f"net_charge_{tag}_mean", f"net_charge_{tag}_se",
                                  "net_charge_pH7_mean", "net_charge_pH7_se")}})
    cc.write_csv(outdir / "pure_chemistries.csv", pure_rows)

    run_info = {
        "task": "three-way composition scan", "stages": stages, "step": args.step,
        "coarse_step": args.coarse_step, "n_compositions": len(points),
        "molecules_per_composition": args.molecules, "bootstrap_resamples": args.bootstrap,
        "seed": args.seed, "weights": args.weights, "target_pH": target, "bands": bands,
        "liming_f_deam": co.LIMING_F_DEAM,
        "pKa": {t.name: v for t, v in base_pka.items()},
        "pKa_ranges": co.PKA_RANGES, "chain_end_models": CHAIN_END_MODELS,
        "native": {"site_counts": {t.name: int(nat[int(t)]) for t in SiteType if nat[int(t)]},
                   "contour_length_nm": length, "mass_Da": mass,
                   "pI": float(co.isoelectric_point_from_counts(nat, base_pka)),
                   f"net_charge_{tag}": float(co.net_charge_from_counts(nat, target, base_pka)),
                   "net_charge_pH7": q7_native},
        "soluble_at_pH7": {"factor_of_native_charge": args.soluble_factor,
                           "threshold_e": threshold},
        "lowest_pI_on_grid": {"pI": float(np.nanmin(pI)),
                              **_composition_row(points[int(np.nanargmin(pI))])},
        "largest_pI_standard_error_within_widest_band": float(max(
            (s["pI_ensemble_se"] for s, p_ in zip(stats, pI)
             if co.in_band(p_, target, bands[-1])), default=float("nan"))),
    }

    # ---- inverse ------------------------------------------------------------
    masks, summary, min_succ = {}, [], []
    if any(s in stages for s in ("inverse", "assays", "stagger")):
        summary, min_succ, masks = solve_inverse(points, pI, q7, target, bands, threshold)
        cc.write_csv(outdir / "region_summary.csv", summary)
        cc.write_csv(outdir / "min_succinylation.csv", min_succ)
        widest = masks[bands[-1]]
        band_of = smallest_band(pI, target, bands)
        edges = np.array([target + sign * b for b in bands for sign in (-1.0, 1.0)])
        region_rows = []
        for i in np.flatnonzero(widest):
            se = stats[i]["pI_ensemble_se"]
            region_rows.append({
                **_composition_row(points[i]), "band": band_of[i],
                "liming_reference": bool(liming[i]),
                "pI_ensemble": pI[i], "pI_ensemble_se": se,
                "within_2se_of_a_band_edge": bool(np.abs(pI[i] - edges).min() < 2.0 * se),
                "pI_molecule_sd": stats[i]["pI_molecule_sd"],
                f"net_charge_{tag}_mean": stats[i][f"net_charge_{tag}_mean"],
                f"net_charge_{tag}_sd": stats[i][f"net_charge_{tag}_sd"],
                f"fraction_net_positive_{tag}": stats[i][f"fraction_net_positive_{tag}"],
                "net_charge_pH7_mean": q7[i],
                "charge_per_nm_pH7_mean": stats[i]["charge_per_nm_pH7_mean"]})
        cc.write_csv(outdir / "consistent_region.csv",
                     [r for r, keep in zip(region_rows, coarse[widest]) if keep])
        cc.write_csv(raw / "consistent_region_fine.csv", region_rows)
        sol_rows = []
        for thr in sorted(set(SOLUBLE_THRESHOLDS_E) | {round(threshold, 3)}):
            for band in bands:
                hit = co.in_band(pI, target, band)
                sol_rows.append({"band": band, "soluble_threshold_e": thr,
                                 "is_default": bool(np.isclose(thr, round(threshold, 3))),
                                 "n_in_band": int(hit.sum()),
                                 "n_consistent": int((hit & co.soluble_at(q7, thr)).sum())})
        cc.write_csv(outdir / "solubility_threshold.csv", sol_rows)
        run_info["inverse"] = {f"band_{b:g}": {k: v for k, v in row.items()}
                               for b, row in zip(bands, summary)}
        print("inverse problem, base parameters:")
        for row in summary:
            reach = {name: ("reaches it" if row[f"pure_{name}_reaches"] else
                            f"stops at pI {row[f'pure_{name}_lowest_pI']:.2f}")
                     for name in ("succinylation", "acetylation", "deamidation")}
            print(f"  +-{row['band']:g}: {row['n_points']} of {len(points)} compositions; "
                  f"alone, succinylation {reach['succinylation']}, acetylation "
                  f"{reach['acetylation']}, deamidation {reach['deamidation']}; least "
                  f"succinylation of any composition {row['f_succ_min']:g}", flush=True)

    # ---- sensitivity ----------------------------------------------------------
    sens_rows, statement_rows, sens_curves = [], [], []
    if "sensitivity" in stages:
        sens_rows, sens_curves = sensitivity(points, sums, native_counts, args.molecules,
                                             target, bands, args.soluble_factor)
        statement_rows = statements(sens_rows, bands)
        cc.write_csv(outdir / "sensitivity.csv", sens_rows)
        cc.write_csv(outdir / "sensitivity_statements.csv", statement_rows)
        cc.write_csv(outdir / "sensitivity_min_succinylation.csv", sens_curves)
        print(f"sensitivity: {len(variants())} variants  ({time.time() - t0:.0f} s)", flush=True)

    # ---- assays -----------------------------------------------------------------
    assay_rows, ranking_rows, set_rows, cand = [], [], [], None
    if "assays" in stages:
        widest = masks[bands[-1]]
        # tabulated: the assay grid plus the liming slice; ranked: the regular
        # grid only, so that every part of composition space counts once
        cand = np.flatnonzero(widest & co.on_grid(points[:, :2], args.assay_step)
                              & (co.on_grid(points[:, 2:], args.assay_step) | liming))
        readouts = co.assay_readouts(mean_counts[cand], nat, mass, base_pka)
        band_of = smallest_band(pI[cand], target, bands)
        for k, i in enumerate(cand):
            assay_rows.append({**_composition_row(points[i]), "band": band_of[k],
                               "liming_reference": bool(liming[i]),
                               **{key: float(v[k]) for key, v in readouts.items()}})
        cc.write_csv(outdir / "assays.csv",
                     [r for r, keep in zip(assay_rows, coarse[cand]) if keep])
        cc.write_csv(raw / "assays_fine.csv", assay_rows)
        keys = [a.key for a in co.ASSAYS]
        for band in bands:
            sel = np.flatnonzero((band_of <= band + 1e-9) & ~liming[cand])
            if len(sel) < 2:
                continue
            sub = {k: v[sel] for k, v in readouts.items()}
            for row in co.rank_assays(points[cand][sel], sub):
                ranking_rows.append(dict(band=band, n_candidates=len(sel), **row))
            for size in (2, 3):
                for row in co.rank_assay_sets(points[cand][sel], sub, keys, size)[:8]:
                    set_rows.append(dict(band=band, n_candidates=len(sel), **row))
        cc.write_csv(outdir / "assay_ranking.csv", ranking_rows)
        cc.write_csv(outdir / "assay_sets.csv", set_rows)
        print(f"assays: {len(cand)} candidate compositions  ({time.time() - t0:.0f} s)",
              flush=True)

    # ---- stagger ------------------------------------------------------------------
    rep_rows, stag_summary, conclusions, crossovers = [], [], [], []
    if "stagger" in stages:
        widest = masks[bands[-1]]
        pool = np.flatnonzero(widest & co.on_grid(points, args.assay_step))
        picked, roles = co.pick_representatives(points[pool], pI[pool], target,
                                                args.representatives)
        chosen = [int(i) for i in pool[picked]]
        # the composition S2 treats as the realistic case, for continuity
        reference = np.flatnonzero(np.all(np.isclose(points, S2_REFERENCE), axis=1))
        if len(reference) and widest[reference[0]]:
            if int(reference[0]) in chosen:
                roles[chosen.index(int(reference[0]))] += ", S2 reference"
            else:
                chosen.append(int(reference[0]))
                roles.append("S2 reference")
        band_of = smallest_band(pI, target, bands)
        for c, i in enumerate(chosen):
            rep_rows.append({"composition": c, **_composition_row(points[i]),
                             "role": roles[c], "band": band_of[i], "pI_ensemble": pI[i],
                             f"net_charge_{tag}_mean": stats[i][f"net_charge_{tag}_mean"],
                             f"net_charge_{tag}_sd": stats[i][f"net_charge_{tag}_sd"],
                             "net_charge_pH7_mean": q7[i],
                             "mean_succinylated": mod["succinyl"][i],
                             "mean_acetylated": mod["acetyl"][i],
                             "mean_deamidated": mod["deamidated"][i]})
        cc.write_csv(outdir / "stagger_representatives.csv", rep_rows)
        ionic = [1e-3 * x for x in args.stagger_ionic_mM]
        orientations = tuple(o for o in args.orientations.split(",") if o)
        stag_summary, minima, pair_rows, gate = stagger_scan(
            native, points[chosen], args.stagger_ph, ionic, args.pairs, args.seed,
            orientations=orientations, weights=accessibility_weights(native, args.weights),
            workers=args.workers, progress=progress("stagger scan, pattern pairs"))
        run_info["stagger"] = {"pH": args.stagger_ph, "ionic_mM": args.stagger_ionic_mM,
                               "pairs": args.pairs, "orientations": orientations,
                               "native_gate": gate}
        if not gate["passed"]:
            print("native gate failed: no stagger results produced", file=sys.stderr)
            cc.write_json(outdir / "run.json", run_info)
            return 2
        cc.write_csv(outdir / "stagger_summary.csv", stag_summary)
        cc.write_csv(raw / "stagger_minima.csv", minima)
        cc.write_csv(raw / "stagger_pairs.csv", pair_rows)
        conclusions = stagger_conclusions(
            stag_summary, native.d_period_nm, target,
            own_pI={r["composition"]: r["pI_ensemble"] for r in rep_rows})
        crossovers = crossover_table(stag_summary)
        cc.write_csv(outdir / "stagger_conclusions.csv", conclusions)
        cc.write_csv(outdir / "stagger_crossover.csv", crossovers)

    # ---- titration curves of the representatives (or of the band extremes) -------
    curve_rows = []
    if "assays" in stages:
        if rep_rows:
            curve_idx = [int(np.flatnonzero(np.all(np.isclose(
                points, (r["f_succ"], r["f_acet"], r["f_deam"])), axis=1))[0]) for r in rep_rows]
        else:
            curve_idx = list(cand[co.pick_representatives(points[cand], pI[cand], target,
                                                          args.representatives)[0]])
        for c, i in enumerate(curve_idx):
            q = co.net_charge_from_counts(mean_counts[i], TITRATION_PH, base_pka)
            for pH, value in zip(TITRATION_PH, q):
                curve_rows.append({"composition": c, **_composition_row(points[i]),
                                   "pH": float(pH), "net_charge_mean": float(value)})
        q = co.net_charge_from_counts(nat, TITRATION_PH, base_pka)
        for pH, value in zip(TITRATION_PH, q):
            curve_rows.append({"composition": -1, "f_succ": 0.0, "f_acet": 0.0, "f_deam": 0.0,
                               "pH": float(pH), "net_charge_mean": float(value)})
        cc.write_csv(outdir / "titration_curves.csv", curve_rows)
        # what each assay would read for those same compositions
        readouts = co.assay_readouts(mean_counts[curve_idx], nat, mass, base_pka)
        cc.write_csv(outdir / "assays_representatives.csv", [
            {"composition": c, **_composition_row(points[i]),
             **{key: float(v[c]) for key, v in readouts.items()}}
            for c, i in enumerate(curve_idx)])

    cc.write_json(outdir / "run.json", run_info)
    if not args.no_figures:
        from . import composition_figures as fig
        fig.write_all(outdir, dict(
            points=points, stats=stats, pI=pI, masks=masks, min_succ=min_succ,
            sens_rows=sens_rows, sens_curves=sens_curves, n_variants=len(variants()),
            assay_rows=assay_rows, ranking_rows=ranking_rows, curve_rows=curve_rows,
            rep_rows=rep_rows, stag_summary=stag_summary, crossovers=crossovers,
            target=target, bands=bands))
    print(f"wrote {outdir}/  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
