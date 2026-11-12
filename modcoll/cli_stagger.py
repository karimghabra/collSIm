"""S2: two-molecule stagger scan without divalent ions.

    python -m modcoll.cli_stagger [--pairs 64] [--seed N] [--outdir DIR] ...

For native collagen and for randomly modified collagen at each modification
fraction, pH and ionic strength, computes the pair energy versus axial
stagger for many independent PAIRS of modification patterns, reduced over the
two facing angles (minimum, Boltzmann average, plain average), and reports
the leading minima, D-selectivity scores and the pair-to-pair spread.

The native control at pH 7.4 is evaluated first, with the repository pKa set
so that it stays comparable with the existing tools; unless it passes the
gate (stagger.native_gate) no modified-collagen result is produced. All
tabulated results, the f = 0 rows included, use --pka-set (default: the
"recommended" set of electrostatics.PKA_SETS).

Outputs (in --outdir): summary.csv, minima.csv, run.json, fig_*.png, and,
under raw/ (large, git-ignored), pairs.csv (per-pair descriptors) and
profiles.npz (mean and spread of every profile).
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import cli_common as cc
from . import electrostatics as es
from . import stagger as st
from .modify import accessibility_weights, modify
from .sites import TELOPEPTIDE_MODELS, native_molecule

WINDOW_NM = 3.0     # "at a D multiple" = within +-3 nm, the README's criterion
LAG_MIN_NM = 5.0    # in-register neighbourhood of zero stagger, excluded

_STATE: dict = {}   # filled before the worker pool forks


def chemistry_fractions(chemistry: str, f: float) -> tuple[float, float]:
    """(f_succ, f_acet) for a total modified amine fraction f."""
    if chemistry == "succ":
        return f, 0.0
    if chemistry == "acet":
        return 0.0, f
    if chemistry == "mixed":
        return 0.5 * f, 0.5 * f
    raise ValueError(f"unknown chemistry {chemistry!r}")


def pattern_rng(seed: int, pair: int, which: int) -> np.random.Generator:
    """Generator of molecule ``which`` (0 = A, 1 = B) of pattern pair ``pair``.
    It does not depend on the modification fraction or on the solution
    conditions, so the same pair index gives nested patterns across fractions
    (see modify.modify) and identical patterns across pH and salt."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(pair, which)))


def make_pair(native, weights, chemistry, f, f_deam, seed, pair):
    f_succ, f_acet = chemistry_fractions(chemistry, f)
    return tuple(modify(native, pattern_rng(seed, pair, k), f_succ=f_succ,
                        f_acet=f_acet, f_deam=f_deam, weights=weights)
                 for k in (0, 1))


def pair_metrics(lags, prof, d_nm) -> dict:
    """Scalar descriptors of one parallel pair's profiles."""
    tot_min, tot_boltz, el_min = prof[0], prof[1], prof[3]
    use = lags > LAG_MIN_NM
    out = {}
    for name, u in (("tot", tot_min), ("el", el_min)):
        i = int(np.argmin(np.where(use, u, np.inf)))
        k, off = st.nearest_multiple(lags[i], d_nm)
        cs = st.comb_selectivity(lags, u, d_nm, WINDOW_NM, lag_min_nm=LAG_MIN_NM)
        out.update({f"{name}_deepest_nm": lags[i], f"{name}_deepest_kT": u[i],
                    f"{name}_deepest_at_multiple": float(k >= 1 and abs(off) <= WINDOW_NM),
                    f"{name}_comb_z": cs["z"], f"{name}_comb_rank": cs["rank"],
                    f"{name}_best_phase_nm": cs["best_phase_nm"]})
    out["boltz_fraction_at_multiple"] = st.boltzmann_multiple_fraction(
        lags, tot_boltz, d_nm, WINDOW_NM, LAG_MIN_NM)
    return out


def _evaluate(f, pair):
    s = _STATE
    model = s["model"]
    mol_a, mol_b = make_pair(s["native"], s["weights"], s["chemistry"], f,
                             s["f_deam"], s["seed"], pair)
    prof = model.profiles(mol_a, mol_b, s["pH"], s["charge_model"])
    d_nm = s["native"].d_period_nm
    stacked, metrics = {}, []
    for o in model.orientations:
        stacked[o] = np.stack([np.stack([prof[(pH, ionic, o)] for ionic in s["ionic"]])
                               for pH in s["pH"]]).astype(np.float32)
    for ip, pH in enumerate(s["pH"]):
        env = es.Environment(pH=pH)
        qa = es.net_charge(mol_a, env, s["charge_model"])
        qb = es.net_charge(mol_b, env, s["charge_model"])
        for ii, ionic in enumerate(s["ionic"]):
            row = {"f": f, "pair": pair, "pH": pH, "ionic_mM": 1e3 * ionic,
                   "net_charge_a": qa, "net_charge_b": qb}
            if "parallel" in stacked:
                row.update(pair_metrics(model.lags["parallel"],
                                        stacked["parallel"][ip, ii].astype(float), d_nm))
            if "antiparallel" in stacked:
                u = stacked["antiparallel"][ip, ii, 0]
                j = int(np.argmin(u))
                row.update({"anti_deepest_nm": model.lags["antiparallel"][j],
                            "anti_deepest_kT": float(u[j])})
            metrics.append(row)
    return f, pair, stacked, metrics


def _worker(task):
    return _evaluate(*task)


def _mean_sd(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def summarise(args, model, d_nm, f_values, mean, sd, metrics):
    """summary.csv and minima.csv rows from the accumulated profiles."""
    chance = st.near_multiple_fraction(model.settings.lag_max_nm, d_nm, WINDOW_NM, LAG_MIN_NM)
    shifts = np.arange(20.0, 140.01, 0.2)
    off_d = np.abs(shifts - d_nm) > 5.0
    summary, minima = [], []
    for i_f, f in enumerate(f_values):
        for ip, pH in enumerate(args.ph):
            for ii, ionic in enumerate(args.ionic):
                rows = [m for m in metrics if m["f"] == f and m["pH"] == pH
                        and np.isclose(m["ionic_mM"], 1e3 * ionic)]
                base = {"chemistry": args.chemistry, "pka_set": args.pka_set, "f": f,
                        "f_deam": args.f_deam, "pH": pH, "ionic_mM": 1e3 * ionic, "n_pairs": len(rows),
                        "net_charge_mean": _mean_sd([m["net_charge_a"] for m in rows]
                                                    + [m["net_charge_b"] for m in rows])[0]}
                for o in model.orientations:
                    lags = model.lags[o]
                    mu, sg = mean[o][i_f, ip, ii], sd[o][i_f, ip, ii]
                    row = dict(base, orientation=o)
                    lag_min = LAG_MIN_NM if o == "parallel" else -1e9
                    lead = []
                    for red in ("tot_min", "tot_boltz", "el_min"):
                        k_red = st.REDUCTIONS.index(red)
                        for rank, (x, v) in enumerate(
                                st.leading_minima(lags, mu[k_red], n=6, lag_min_nm=lag_min), 1):
                            j = int(np.argmin(np.abs(lags - x)))
                            k, off = st.nearest_multiple(x, d_nm)
                            minima.append(dict(base, orientation=o, profile=red, rank=rank,
                                               stagger_nm=x, stagger_D=x / d_nm,
                                               nearest_multiple=k, offset_nm=off,
                                               energy_mean_kT=v, energy_pair_sd_kT=sg[k_red][j]))
                            if red == "tot_min":
                                lead.append(minima[-1])
                    if lead:
                        row.update({"deepest_nm": lead[0]["stagger_nm"],
                                    "deepest_D": lead[0]["stagger_D"],
                                    "deepest_kT": lead[0]["energy_mean_kT"],
                                    "deepest_pair_sd_kT": lead[0]["energy_pair_sd_kT"],
                                    "top5_D": " ".join(f"{m['stagger_D']:.2f}" for m in lead[:5])})
                    if o == "parallel":
                        cs = st.comb_selectivity(lags, mu[0], d_nm, WINDOW_NM,
                                                 lag_min_nm=LAG_MIN_NM)
                        cs_el = st.comb_selectivity(lags, mu[3], d_nm, WINDOW_NM,
                                                    lag_min_nm=LAG_MIN_NM)
                        ac = st.profile_autocorrelation(lags, mu[0], shifts, LAG_MIN_NM)
                        period, period_r = st.dominant_period(shifts, ac)
                        ac_d = ac[int(np.argmin(np.abs(shifts - d_nm)))]
                        row.update({
                            "deepest_at_multiple": bool(lead and lead[0]["nearest_multiple"] >= 1
                                                        and abs(lead[0]["offset_nm"]) <= WINDOW_NM),
                            "comb_z_mean_profile": cs["z"], "comb_rank_mean_profile": cs["rank"],
                            "comb_best_phase_nm": cs["best_phase_nm"],
                            "el_comb_z_mean_profile": cs_el["z"],
                            # self-similarity of the mean profile under a shift
                            "autocorr_at_D": ac_d,
                            "autocorr_at_D_z": (ac_d - np.nanmean(ac[off_d]))
                            / np.nanstd(ac[off_d]),
                            "dominant_period_nm": period, "dominant_period_r": period_r,
                            "chance_fraction_at_multiple": chance})
                        for key, name in (("tot_comb_z", "comb_z_pairs"),
                                          ("el_comb_z", "el_comb_z_pairs"),
                                          ("tot_deepest_kT", "deepest_kT_pairs"),
                                          ("tot_deepest_at_multiple", "pairs_deepest_at_multiple"),
                                          ("boltz_fraction_at_multiple", "boltz_fraction_at_multiple")):
                            m_, s_ = _mean_sd([m[key] for m in rows])
                            row[f"{name}_mean"], row[f"{name}_sd"] = m_, s_
                    else:
                        m_, s_ = _mean_sd([m["anti_deepest_kT"] for m in rows])
                        row["deepest_kT_pairs_mean"], row["deepest_kT_pairs_sd"] = m_, s_
                        if "tot_deepest_kT" in rows[0]:
                            m_, s_ = _mean_sd([m["anti_deepest_kT"] - m["tot_deepest_kT"]
                                               for m in rows])
                            row["anti_minus_parallel_kT_mean"] = m_
                            row["anti_minus_parallel_kT_sd"] = s_
                    summary.append(row)
    return summary, minima


# ------------------------------------------------------------------ figures ---
def _mark_multiples(ax, d_nm, lag_max):
    for k in range(1, int(lag_max / d_nm) + 1):
        ax.axvline(k * d_nm, color=cc.INK_SECONDARY, lw=0.7, ls=(0, (2, 3)), zorder=1)


def figure_gate(path, lags, prof, gate, d_nm):
    plt = cc.pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.6), sharex=True)
    panels = (("total energy, minimum over facing angles", prof[0], gate["min_over_angles_minima"]),
              ("total energy, averaged over facing angles", prof[2], gate["azimuth_mean_minima"]),
              ("electrostatic term alone, minimum over facing angles", prof[3], None))
    for ax, (title, u, mins) in zip(axes, panels):
        show = lags > LAG_MIN_NM
        ax.plot(lags[show], u[show], color=cc.CATEGORICAL[0], lw=1.2)
        _mark_multiples(ax, d_nm, lags[-1])
        if mins is None:
            mins = [{"stagger_nm": x, "stagger_D": x / d_nm, "energy_kT": v}
                    for x, v in st.leading_minima(lags, u, n=5)]
        for m in mins:
            ax.plot(m["stagger_nm"], m["energy_kT"], "o", color=cc.INK, ms=4, zorder=3)
            ax.annotate(f"{m['stagger_D']:.2f} D", (m["stagger_nm"], m["energy_kT"]),
                        textcoords="offset points", xytext=(0, -11), ha="center",
                        fontsize=8, color=cc.INK)
        lo, hi = u[show].min(), u[show].max()
        ax.set_ylim(lo - 0.16 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(title)
        ax.set_ylabel("pair energy (kT)")
    axes[-1].set_xlabel("axial stagger (nm); dashed lines = integer multiples of D")
    rect = cc.headline(fig, "Native control: parallel pair, pH 7.4, 150 mM "
                       f"(gate {'passed' if gate['passed'] else 'FAILED'})",
                       "dots = five deepest minima of each profile")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_profiles(path, lags, mean, sd, f_values, ph_values, ionic_mM, d_nm, chemistry):
    plt = cc.pyplot()
    fig, axes = plt.subplots(len(ph_values), len(f_values), sharex=True, sharey="row",
                             figsize=(2.35 * len(f_values) + 0.6, 1.75 * len(ph_values) + 0.9),
                             squeeze=False)
    show = lags > LAG_MIN_NM
    for r, pH in enumerate(ph_values):
        for c, f in enumerate(f_values):
            ax = axes[r, c]
            mu, sg = mean[c, r][0][show], sd[c, r][0][show]
            ax.fill_between(lags[show], mu - sg, mu + sg, color=cc.CATEGORICAL[1],
                            lw=0, alpha=0.45)
            ax.plot(lags[show], mu, color=cc.ORDINAL_BLUE[3], lw=0.6)
            _mark_multiples(ax, d_nm, lags[-1])
            j = int(np.argmin(mu))
            ax.plot(lags[show][j], mu[j], "o", color=cc.INK, ms=3.5, zorder=3)
            ax.annotate(f"{lags[show][j] / d_nm:.2f} D", (lags[show][j], mu[j]),
                        textcoords="offset points", xytext=(5, -2), fontsize=7.5)
            if r == 0:
                ax.set_title("native (f = 0)" if f == 0 else f"f = {f:g}")
            if c == 0:
                ax.set_ylabel(f"pH {pH:g}\nenergy (kT)")
            if r == len(ph_values) - 1:
                ax.set_xlabel("stagger (nm)")
    rect = cc.headline(fig, "Pair energy vs stagger, minimum over facing angles "
                       f"({chemistry}, {ionic_mM:g} mM)",
                       "line = mean over pattern pairs, band = pair-to-pair sd, "
                       "dot = deepest minimum, dashed = integer multiples of D")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_selectivity(path, summary, f_values, ph_values, ionic_values, chance):
    plt = cc.pyplot()
    par = [r for r in summary if r["orientation"] == "parallel"]
    quantities = (
        ("comb_z_pairs", "D-comb z-score\n(per pair)", None),
        ("deepest_kT_pairs", "deepest minimum\n(kT, per pair)", None),
        ("pairs_deepest_at_multiple", "pairs with deepest\nminimum at k·D", chance),
    )
    fig, axes = plt.subplots(len(quantities), len(ionic_values), sharex=True, sharey="row",
                             figsize=(2.7 * len(ionic_values) + 0.8, 2.1 * len(quantities) + 0.8),
                             squeeze=False)
    for r, (key, label, ref) in enumerate(quantities):
        for c, ionic in enumerate(ionic_values):
            ax = axes[r, c]
            for k, pH in enumerate(ph_values):
                rows = sorted((x for x in par if x["pH"] == pH
                               and np.isclose(x["ionic_mM"], 1e3 * ionic)),
                              key=lambda x: x["f"])
                x = np.array([row["f"] for row in rows]) + 0.012 * (k - 1.5)
                y = np.array([row[f"{key}_mean"] for row in rows])
                e = np.array([row[f"{key}_sd"] for row in rows])
                if key == "pairs_deepest_at_multiple":
                    e = None                 # a proportion: sd of 0/1 is not shown
                col = cc.CATEGORICAL[k % len(cc.CATEGORICAL)]
                ax.errorbar(x, y, yerr=e, color=col, marker=cc.MARKERS[k % 4], ms=4,
                            lw=1.3, elinewidth=0.8, capsize=0, label=f"pH {pH:g}")
            if ref is not None:
                ax.axhline(ref, color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)))
                if c == 0:
                    ax.annotate("chance", (0.0, ref), textcoords="offset points",
                                xytext=(2, 3), fontsize=7.5, color=cc.INK_SECONDARY)
            if key == "comb_z_pairs":
                ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
            if r == 0:
                ax.set_title(f"{1e3 * ionic:g} mM")
            if c == 0:
                ax.set_ylabel(label)
            if r == len(quantities) - 1:
                ax.set_xlabel("modified fraction f")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.6)
    rect = cc.headline(fig, "Registry selectivity vs modified fraction",
                       "points = mean over pattern pairs, bars = pair-to-pair sd")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_positions(path, metrics, f_values, ph_values, ionic, d_nm, seed):
    plt = cc.pyplot()
    fig, axes = plt.subplots(1, len(ph_values), sharey=True, squeeze=False,
                             figsize=(2.6 * len(ph_values) + 0.8, 4.2))
    jitter = np.random.default_rng(seed)       # display jitter only
    for ax, pH in zip(axes[0], ph_values):
        for i, f in enumerate(f_values):
            y = np.array([m["tot_deepest_nm"] / d_nm for m in metrics
                          if m["f"] == f and m["pH"] == pH
                          and np.isclose(m["ionic_mM"], 1e3 * ionic)])
            x = i + (jitter.random(len(y)) - 0.5) * 0.55 if len(y) > 1 else np.full(len(y), i)
            ax.plot(x, y, "o", color=cc.CATEGORICAL[0], ms=3.2, alpha=0.55, mew=0)
        for k in range(1, 5):
            ax.axhline(k, color=cc.INK_SECONDARY, lw=0.7, ls=(0, (2, 3)), zorder=1)
        ax.set_xticks(range(len(f_values)))
        ax.set_xticklabels([f"{f:g}" for f in f_values])
        ax.set_title(f"pH {pH:g}")
        ax.set_xlabel("modified fraction f")
        ax.grid(False)
    axes[0, 0].set_ylabel("stagger of each pair's deepest minimum (units of D)")
    rect = cc.headline(fig, f"Where each pattern pair binds best ({1e3 * ionic:g} mM)",
                       "one dot per pattern pair; dashed = integer multiples of D")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------- main ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--f", type=cc.float_list, default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
                   help="modified amine fractions (0 = native control)")
    p.add_argument("--ph", type=cc.float_list, default=[3.5, 4.2, 5.0, 7.4])
    p.add_argument("--ionic-mM", dest="ionic_mM", type=cc.float_list,
                   default=[10.0, 50.0, 150.0, 300.0],
                   help="ionic strength of the monovalent background, mmol/L")
    p.add_argument("--chemistry", choices=("succ", "acet", "mixed"), default="succ")
    p.add_argument("--f-deam", type=float, default=0.0,
                   help="deamidated fraction of Asn/Gln, applied at every f")
    p.add_argument("--pka-set", choices=sorted(es.PKA_SETS), default=es.DEFAULT_PKA_SET,
                   help="pKa set of the tabulated results (the native gate always "
                        "uses the repository set)")
    p.add_argument("--weights", choices=("uniform", "crosslink_sites", "ion_pair"),
                   default="uniform",
                   help="site reactivity model (modify.accessibility_weights)")
    p.add_argument("--pairs", type=int, default=64, help="independent pattern pairs per f")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--orientations", default="parallel,antiparallel")
    p.add_argument("--telopeptides", choices=TELOPEPTIDE_MODELS, default="mature")
    p.add_argument("--position", choices=("centroid", "group"), default="centroid")
    p.add_argument("--gap-nm", type=float, default=st.PairSettings.surface_separation_nm,
                   help="surface separation of the two molecules, nm")
    p.add_argument("--eps-hydrophobic", type=float, default=es.EPS_HYDROPHOBIC)
    p.add_argument("--core-nm", type=float, default=es.SOFT_CORE_NM)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--outdir", default=None)
    p.add_argument("--gate-only", action="store_true",
                   help="evaluate the native gate, write it and stop")
    p.add_argument("--allow-failed-gate", action="store_true",
                   help="produce modified-collagen results even if the native gate fails")
    p.add_argument("--no-figures", action="store_true")
    return p


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.ionic = [1e-3 * x for x in args.ionic_mM]
    orientations = tuple(o for o in args.orientations.split(",") if o)
    outdir = Path(args.outdir or f"results/modcoll/stagger_{args.chemistry}")
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    native = native_molecule(args.telopeptides)
    d_nm = native.d_period_nm
    settings = st.PairSettings(surface_separation_nm=args.gap_nm, core_nm=args.core_nm,
                               eps_hydrophobic=args.eps_hydrophobic, position=args.position)

    # ---- native gate: parallel pair, pH 7.4, 150 mM, repository pKa set -------
    charge_model = es.charge_model_for(args.pka_set)
    gate_model = st.PairModel(native, [0.15], settings, ("parallel",))
    gate_lags = gate_model.lags["parallel"]
    gate_prof = gate_model.profiles(native, native, [7.4], es.charge_model_for(
        "repository"))[(7.4, 0.15, "parallel")]
    gate = st.native_gate(gate_lags, gate_prof[0], gate_prof[2], d_nm)
    gate["pka_set"] = "repository"
    # the same control with the pKa set of the results, for the record only
    own_prof = gate_model.profiles(native, native, [7.4], charge_model)[(7.4, 0.15, "parallel")]
    gate_own = st.native_gate(gate_lags, own_prof[0], own_prof[2], d_nm)
    gate_own["pka_set"] = args.pka_set
    print(f"native gate (pH 7.4, 150 mM, repository pKa set): "
          f"{'PASSED' if gate['passed'] else 'FAILED'}")
    for name in ("min_over_angles_minima", "azimuth_mean_minima"):
        print(f"  {name}: " + "  ".join(f"{m['stagger_D']:.2f} D ({m['energy_kT']:+.1f} kT)"
                                         for m in gate[name]))
    if not args.no_figures:
        figure_gate(outdir / "fig_native_gate.png", gate_lags, gate_prof, gate, d_nm)
    run_info = {
        "task": "S2 two-molecule stagger scan", "chemistry": args.chemistry,
        "f": args.f, "f_deam": args.f_deam, "pH": args.ph, "ionic_mM": args.ionic_mM,
        "weights": args.weights, "pairs": args.pairs, "seed": args.seed,
        "pka_set": args.pka_set, "pKa": {t.name: v for t, v in es.PKA_SETS[args.pka_set].items()},
        "orientations": orientations, "telopeptides": args.telopeptides,
        "settings": {"axis_separation_nm": settings.axis_separation_nm,
                     "surface_separation_nm": settings.surface_separation_nm,
                     "soft_core_nm": settings.core_nm,
                     "eps_hydrophobic_kT": settings.eps_hydrophobic,
                     "hydrophobic_sigma_nm": settings.hydrophobic_sigma_nm,
                     "charge_position": settings.position,
                     "bjerrum_nm": es.BJERRUM_NM,
                     "stagger_step_nm": settings.grid.ds_nm,
                     "facing_angles": settings.grid.n_phi,
                     "window_nm": WINDOW_NM, "lag_min_nm": LAG_MIN_NM},
        "d_period_nm": d_nm, "helix_length_nm": native.helix_length_nm,
        "n_sites": len(native), "native_gate": gate,
        "native_control_with_result_pka_set": gate_own,
    }
    cc.write_json(outdir / "run.json", run_info)
    if args.gate_only:
        return 0 if gate["passed"] else 2
    if not gate["passed"] and not args.allow_failed_gate:
        print("native gate failed: no modified-collagen results produced "
              "(--allow-failed-gate overrides)", file=sys.stderr)
        return 2

    # ---- the scan -------------------------------------------------------------
    model = st.PairModel(native, args.ionic, settings, orientations)
    weights = accessibility_weights(native, args.weights)
    _STATE.update(model=model, native=native, weights=weights, chemistry=args.chemistry,
                  f_deam=args.f_deam, seed=args.seed, pH=args.ph, ionic=args.ionic,
                  charge_model=charge_model)
    tasks = []
    for f in args.f:
        random = (f > 0.0) or (args.f_deam > 0.0)
        tasks += [(f, pair) for pair in range(args.pairs if random else 1)]
    shape = (len(args.f), len(args.ph), len(args.ionic), len(st.REDUCTIONS))
    acc = {o: np.zeros(shape + (len(model.lags[o]),)) for o in orientations}
    acc2 = {o: np.zeros(shape + (len(model.lags[o]),)) for o in orientations}
    count = np.zeros(len(args.f), int)
    metrics = []

    def consume(result):
        f, pair, stacked, rows = result
        i_f = args.f.index(f)
        for o in orientations:
            x = stacked[o].astype(float)
            acc[o][i_f] += x
            acc2[o][i_f] += x * x
        count[i_f] += 1
        metrics.extend(rows)

    if args.workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("fork")   # workers share the kernel tables
        with ctx.Pool(args.workers) as pool:
            for k, result in enumerate(pool.imap_unordered(_worker, tasks), 1):
                consume(result)
                if k % 32 == 0 or k == len(tasks):
                    print(f"  {k}/{len(tasks)} pattern pairs  ({time.time() - t0:.0f} s)",
                          flush=True)
    else:
        for task in tasks:
            consume(_evaluate(*task))
    metrics.sort(key=lambda m: (m["f"], m["pair"], m["pH"], m["ionic_mM"]))

    mean, sd = {}, {}
    for o in orientations:
        n = count[:, None, None, None, None]
        mean[o] = acc[o] / n
        var = np.clip(acc2[o] / n - mean[o] ** 2, 0.0, None) * (n / np.maximum(n - 1, 1))
        sd[o] = np.where(n > 1, np.sqrt(var), 0.0)

    summary, minima = summarise(args, model, d_nm, args.f, mean, sd, metrics)
    cc.write_csv(outdir / "summary.csv", summary)
    cc.write_csv(outdir / "minima.csv", minima)
    raw = outdir / "raw"
    raw.mkdir(exist_ok=True)
    cc.write_csv(raw / "pairs.csv",
                 [dict(chemistry=args.chemistry, **m) for m in metrics])
    np.savez_compressed(
        raw / "profiles.npz", f=np.array(args.f), pH=np.array(args.ph),
        ionic_mM=np.array(args.ionic_mM), reductions=np.array(st.REDUCTIONS),
        **{f"lags_{o}": model.lags[o] for o in orientations},
        **{f"mean_{o}": mean[o].astype(np.float32) for o in orientations},
        **{f"sd_{o}": sd[o].astype(np.float32) for o in orientations})

    if not args.no_figures and "parallel" in orientations:
        i_ref = int(np.argmin(np.abs(np.array(args.ionic) - 0.15)))
        figure_profiles(outdir / "fig_profiles_vs_f.png", model.lags["parallel"],
                        mean["parallel"][:, :, i_ref], sd["parallel"][:, :, i_ref],
                        args.f, args.ph, args.ionic_mM[i_ref], d_nm, args.chemistry)
        chance = st.near_multiple_fraction(settings.lag_max_nm, d_nm, WINDOW_NM, LAG_MIN_NM)
        figure_selectivity(outdir / "fig_selectivity_vs_f.png", summary, args.f, args.ph,
                           args.ionic, chance)
        figure_positions(outdir / "fig_deepest_minimum_positions.png", metrics, args.f,
                         args.ph, args.ionic[i_ref], d_nm, args.seed)
    print(f"wrote {outdir}/  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
