"""S1: statistics of the randomly modified ensemble.

    python -m modcoll.cli_ensemble [--molecules 256] [--seed N] [--outdir DIR]

For each chemistry and modification fraction f, samples independent molecules
(each with its own quenched pattern) and reports the net-charge distribution
at each pH, the isoelectric point, and the correlation of the axial charge
pattern at the D period (234 residues), within one molecule and between two
independently modified molecules, for the signed charge and for its magnitude
(where the charged residues are, regardless of sign).

Chemistries: succ (succinylation), acet (acetylation), mixed (half of the
modified amines each), deam (deamidation of Asn/Gln only), succ_crosslink
(succinylation with the "crosslink_sites" reactivity weights: the six
cross-linking lysines react ten times faster).

Outputs (in --outdir): pi_vs_f.csv, net_charge.csv (distribution between
molecules: mean, sd, quantiles, share of net-positive molecules),
net_charge_histogram.csv, titration.csv, d_lag_correlation.csv,
autocorrelation_profile.csv, gates.json, fig_*.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import cli_common as cc
from . import electrostatics as es
from . import stagger as st
from .modify import accessibility_weights, modify
from .sites import D_PERIOD_RESIDUES, TELOPEPTIDE_MODELS, SiteType, native_molecule

CHEMISTRIES = ("succ", "acet", "mixed", "deam", "succ_crosslink")
PROFILE_LAG_MIN = 150   # first lag (residues) written to the profile table
HIST_BIN_E = 2.0        # bin width of the net-charge histogram, elementary charges

LABELS = {"succ": "succinylation", "acet": "acetylation",
          "mixed": "succinyl + acetyl (1:1)", "deam": "deamidation only",
          "succ_crosslink": "succinylation, cross-link sites 10x"}


def fractions(chemistry: str, f: float) -> dict:
    """modify() keyword arguments for one chemistry at fraction f."""
    if chemistry in ("succ", "succ_crosslink"):
        return {"f_succ": f}
    if chemistry == "acet":
        return {"f_acet": f}
    if chemistry == "mixed":
        return {"f_succ": 0.5 * f, "f_acet": 0.5 * f}
    if chemistry == "deam":
        return {"f_deam": f}
    raise ValueError(f"unknown chemistry {chemistry!r}")


def sample(native, chemistry: str, f: float, n: int, seed: int, f_deam: float = 0.0):
    """``n`` independently modified molecules. Molecule k of a chemistry uses
    the same generator state at every f, so patterns are nested across f."""
    weights = accessibility_weights(native, "crosslink_sites"
                                    if chemistry == "succ_crosslink" else "uniform")
    kw = fractions(chemistry, f)
    if chemistry != "deam":
        kw["f_deam"] = f_deam
    chem_id = CHEMISTRIES.index(chemistry.split("_")[0])   # same deviates for both weightings
    return [modify(native, np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(chem_id, k))), weights=weights, **kw)
        for k in range(n)]


def d_lag_statistics(mols, env, lags, charge_model=es.henderson_hasselbalch):
    """Lagged correlation of the axial charge pattern, per molecule (self)
    and per neighbour pair of independent patterns (cross; pair k is molecule
    k with molecule k+1, cyclically), at each lag in residues.

    Two patterns are analysed: "signed", the charge itself, and "abs", its
    magnitude, i.e. where the charged residues are regardless of sign.
    Returns {pattern: (self array [n, n_lag], cross array [n, n_lag])}.
    """
    out = {}
    charges = [charge_model(m, env) for m in mols]
    for name, fn in (("signed", lambda q: q), ("abs", np.abs)):
        # all molecules share one geometry, hence one set of axial bins
        profs = np.array([st.axial_profile(m, fn(q))[1] for m, q in zip(mols, charges)])
        partner = np.roll(profs, -1, axis=0)
        out[name] = (st.lagged_correlation_rows(profs, profs, lags),
                     st.lagged_correlation_rows(profs, partner, lags))
    return out


def _sd(x) -> float:
    """Sample standard deviation; exactly 0 for identical values (the mean of
    equal floats is not exact, which would otherwise leave 1e-15 noise)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.ptp(x) == 0.0:
        return 0.0
    return float(np.std(x, ddof=1))


# ------------------------------------------------------------------ figures ---
def figure_pi(path, rows, target_pH):
    plt = cc.pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    shown = ("succ", "acet", "mixed", "deam")
    for k, chem in enumerate(shown):
        sel = sorted((r for r in rows if r["chemistry"] == chem), key=lambda r: r["f"])
        if not sel:
            continue
        x = np.array([r["f"] for r in sel])
        y = np.array([r["pI_ensemble"] for r in sel])
        e = np.array([r["pI_molecule_sd"] for r in sel])
        col = cc.CATEGORICAL[k]
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.18, lw=0)
        ax.plot(x, y, color=col, marker=cc.MARKERS[k], ms=4, label=LABELS[chem])
    ax.axhline(target_pH, color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)))
    ax.annotate(f"pH {target_pH:g}", (0.0, target_pH), textcoords="offset points",
                xytext=(2, 4), fontsize=8, color=cc.INK_SECONDARY)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("modified fraction f (of amines; of Asn/Gln for deamidation)")
    ax.set_ylabel("isoelectric point (pH)\nline = ensemble, band = molecule-to-molecule sd")
    ax.set_title("Isoelectric point of randomly modified type I collagen")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_net_charge(path, titration, charge_rows, f_show, ph_values):
    plt = cc.pyplot()
    chems = [c for c in ("succ", "acet", "mixed") if any(r["chemistry"] == c for r in titration)]
    fig, axes = plt.subplots(2, len(chems), figsize=(3.5 * len(chems) + 0.8, 6.6),
                             sharey="row", squeeze=False)
    cols = cc.ordinal_colors(max(len(f_show) - 1, 1))
    for c, chem in enumerate(chems):
        ax = axes[0, c]
        for i, f in enumerate(f_show):
            sel = sorted((r for r in titration if r["chemistry"] == chem
                          and np.isclose(r["f"], f)), key=lambda r: r["pH"])
            if not sel:
                continue
            col = cc.INK if i == 0 else cols[i - 1]
            ax.plot([r["pH"] for r in sel], [r["net_charge_mean"] for r in sel], color=col,
                    lw=1.4, ls="--" if i == 0 else "-",
                    label="native (f = 0)" if i == 0 else f"f = {f:g}")
        ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.set_title(LABELS[chem])
        ax.set_xlabel("pH")
        if c == 0:
            ax.set_ylabel("mean net charge per molecule (e)")
        ax = axes[1, c]
        for k, pH in enumerate(ph_values):
            sel = sorted((r for r in charge_rows if r["chemistry"] == chem
                          and np.isclose(r["pH"], pH)), key=lambda r: r["f"])
            x = np.array([r["f"] for r in sel])
            y = np.array([r["net_charge_mean"] for r in sel])
            e = np.array([r["net_charge_sd"] for r in sel])
            col = cc.CATEGORICAL[k % 4]
            ax.errorbar(x, y, yerr=e, color=col, marker=cc.MARKERS[k % 4], ms=3.5, lw=1.2,
                        elinewidth=0.8, label=f"pH {pH:g}")
        ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.set_xlabel("modified fraction f")
        if c == 0:
            ax.set_ylabel("net charge per molecule (e)\nmean, bars = molecule-to-molecule sd")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.8)
    axes[1, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.8)
    fig.tight_layout(rect=cc.headline(fig, "Net charge of randomly modified type I collagen"))
    fig.savefig(path)
    plt.close(fig)


def figure_charge_distribution(path, hist_rows, f_show, pH):
    """Histogram of the net charge of individual molecules at one pH: one
    panel per modified fraction, one colour per chemistry."""
    if not f_show:
        return
    plt = cc.pyplot()
    chems = [c for c in ("succ", "acet", "mixed") if any(r["chemistry"] == c for r in hist_rows)]
    fig, axes = plt.subplots(1, len(f_show), figsize=(3.3 * len(f_show) + 0.6, 3.6),
                             sharey=True, squeeze=False)
    for ax, f in zip(axes[0], f_show):
        for k, chem in enumerate(chems):
            sel = [r for r in hist_rows if r["chemistry"] == chem and np.isclose(r["f"], f)]
            ax.bar([r["bin_from_e"] for r in sel], [r["molecules"] for r in sel],
                   width=HIST_BIN_E * 0.9, align="edge", color=cc.CATEGORICAL[k],
                   alpha=0.75, lw=0, label=LABELS[chem])
        ax.axvline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.set_title(f"f = {f:g}")
        ax.set_xlabel("net charge of a molecule (e)")
    axes[0, 0].set_ylabel("molecules")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout(rect=cc.headline(
        fig, f"Net charge of individual molecules at pH {pH:g}",
        "each molecule has its own random modification pattern"))
    fig.savefig(path)
    plt.close(fig)


def figure_autocorrelation(path, profile_rows, dlag_rows, pH):
    """Rows: the two patterns (signed charge, charge magnitude). Left: the
    correlation against lag for three ensembles; right: its value at lag D
    against the modified fraction."""
    plt = cc.pyplot()
    patterns = (("signed", "signed charge q"), ("abs", "charge magnitude |q|"))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6), squeeze=False)
    curves = (("succ", 0.0, "self_mean", cc.INK, "native"),
              ("succ", 1.0, "self_mean", cc.CATEGORICAL[0], "succinylated, f = 1"),
              ("succ", 0.5, "cross_mean", cc.CATEGORICAL[1],
               "succinylated, f = 0.5 (two molecules)"))
    for r, (pat, pat_label) in enumerate(patterns):
        ax = axes[r, 0]
        for chem, f, key, col, lab in curves:
            sel = sorted((x for x in profile_rows if x["chemistry"] == chem
                          and np.isclose(x["f"], f) and np.isclose(x["pH"], pH)
                          and x["pattern"] == pat), key=lambda x: x["lag_residues"])
            if sel:
                ax.plot([x["lag_residues"] for x in sel], [x[key] for x in sel],
                        color=col, lw=1.0, label=lab)
        ax.axvline(D_PERIOD_RESIDUES, color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)))
        ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.set_title(f"{pat_label}: correlation vs lag")
        ax.set_ylabel("Pearson correlation")
        if r == 1:
            ax.set_xlabel("lag (residues); dashed = D = 234")
        ax = axes[r, 1]
        for k, chem in enumerate(("succ", "acet", "mixed")):
            sel = sorted((x for x in dlag_rows if x["chemistry"] == chem
                          and np.isclose(x["pH"], pH)), key=lambda x: x["f"])
            if not sel:
                continue
            x = np.array([row["f"] for row in sel]) + 0.008 * (k - 1)
            ax.errorbar(x, [row[f"{pat}_cross_r_mean"] for row in sel],
                        yerr=[row[f"{pat}_cross_r_sd"] for row in sel],
                        color=cc.CATEGORICAL[k], marker=cc.MARKERS[k], ms=4, lw=1.3,
                        elinewidth=0.8, label=LABELS[chem])
        ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.set_title(f"{pat_label}: two independent molecules, lag D")
        ax.set_ylabel("correlation at 234 residues")
        if r == 1:
            ax.set_xlabel("modified fraction f (bars = sd over molecule pairs)")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, ncols=1)
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.tight_layout(rect=cc.headline(
        fig, f"D-periodicity of the axial charge pattern, pH {pH:g}"))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------- main ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--f", type=cc.float_list,
                   default=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
                            0.95, 0.97, 1.0])
    p.add_argument("--ph", type=cc.float_list, default=[3.5, 4.2, 5.0, 7.4])
    p.add_argument("--chemistries", default=",".join(CHEMISTRIES))
    p.add_argument("--f-deam", type=float, default=0.0,
                   help="deamidated fraction applied on top of the amine chemistries")
    p.add_argument("--molecules", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--telopeptides", choices=TELOPEPTIDE_MODELS, default="mature")
    p.add_argument("--pka-set", choices=sorted(es.PKA_SETS), default=es.DEFAULT_PKA_SET)
    p.add_argument("--ionic-mM", dest="ionic_mM", type=float, default=150.0,
                   help="recorded with the results; Henderson-Hasselbalch charges "
                        "do not depend on it")
    p.add_argument("--target-pH", type=float, default=4.2)
    p.add_argument("--outdir", default="results/modcoll/ensemble")
    p.add_argument("--no-figures", action="store_true")
    return p


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    t0 = time.time()
    native = native_molecule(args.telopeptides)
    chems = [c for c in args.chemistries.split(",") if c]
    ionic = 1e-3 * args.ionic_mM
    charge_model = es.charge_model_for(args.pka_set)
    hist_rows = []
    ph_grid = np.round(np.arange(2.0, 12.01, 0.1), 2)
    corr_ph = [p for p in (4.2, 7.4) if p in args.ph] or args.ph[:1]
    lags = np.arange(0, 301)
    f_profile = [f for f in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0) if f in args.f]
    off_d = np.array([l for l in range(30, 301) if abs(l - D_PERIOD_RESIDUES) > 5])

    pi_rows, charge_rows, titration, dlag_rows, profile_rows = [], [], [], [], []
    for chem in chems:
        for f in args.f:
            mols = sample(native, chem, f, args.molecules, args.seed, args.f_deam)
            pis = np.array([es.isoelectric_point(m, ionic, charge_model) for m in mols])
            counts = {t.name: float(np.mean([m.count(t) for m in mols])) for t in SiteType}
            pi_rows.append({
                "chemistry": chem, "f": f, "n_molecules": len(mols),
                "pI_ensemble": es.isoelectric_point(mols, ionic, charge_model),
                "pI_molecule_mean": pis.mean(), "pI_molecule_sd": _sd(pis),
                "pI_molecule_min": pis.min(), "pI_molecule_max": pis.max(),
                "mean_succinylated": counts["SUCC_LYS"] + counts["SUCC_NTERM"],
                "mean_acetylated": counts["ACET_LYS"] + counts["ACET_NTERM"],
                "mean_unmodified_amines": counts["LYS"] + counts["NTERM"],
                "mean_amides_left": counts["ASN"] + counts["GLN"]})
            for pH in args.ph:
                q = np.array([es.net_charge(m, es.Environment(pH, ionic), charge_model) for m in mols])
                p5, p50, p95 = np.percentile(q, [5, 50, 95])
                charge_rows.append({"chemistry": chem, "f": f, "pH": pH,
                                    "n_molecules": len(mols), "net_charge_mean": q.mean(),
                                    "net_charge_sd": _sd(q), "net_charge_min": q.min(),
                                    "net_charge_p05": p5, "net_charge_median": p50,
                                    "net_charge_p95": p95, "net_charge_max": q.max(),
                                    # a sample at its mean isoelectric point still
                                    # holds molecules of both signs
                                    "fraction_net_positive": float(np.mean(q > 0))})
                if np.isclose(pH, args.target_pH):
                    lo = HIST_BIN_E * np.floor(q.min() / HIST_BIN_E)
                    edges = np.arange(lo, q.max() + HIST_BIN_E, HIST_BIN_E)
                    counts, edges = np.histogram(q, bins=max(len(edges) - 1, 1),
                                                 range=(lo, lo + HIST_BIN_E * max(len(edges) - 1, 1)))
                    for left, n_in in zip(edges[:-1], counts):
                        hist_rows.append({"chemistry": chem, "f": f, "pH": pH,
                                          "bin_from_e": left, "bin_to_e": left + HIST_BIN_E,
                                          "molecules": int(n_in)})
            if f in f_profile:
                for pH in ph_grid:
                    env = es.Environment(float(pH), ionic)
                    titration.append({"chemistry": chem, "f": f, "pH": float(pH),
                                      "net_charge_mean": float(np.mean(
                                          [es.net_charge(m, env, charge_model) for m in mols]))})
            for pH in corr_ph:
                stats = d_lag_statistics(mols, es.Environment(pH, ionic), lags,
                                         charge_model)
                d = D_PERIOD_RESIDUES
                row = {"chemistry": chem, "f": f, "pH": pH, "n_molecules": len(mols)}
                for name, (self_r, cross_r) in stats.items():
                    sm, cm = self_r.mean(axis=0), cross_r.mean(axis=0)
                    row.update({
                        f"{name}_self_r_mean": sm[d], f"{name}_self_r_sd": _sd(self_r[:, d]),
                        f"{name}_cross_r_mean": cm[d], f"{name}_cross_r_sd": _sd(cross_r[:, d]),
                        # how far lag D stands out from the other lags 30..300
                        f"{name}_self_z_vs_other_lags":
                            (sm[d] - sm[off_d].mean()) / sm[off_d].std(),
                        f"{name}_cross_z_vs_other_lags":
                            (cm[d] - cm[off_d].mean()) / cm[off_d].std(),
                        f"{name}_cross_lag_of_maximum": int(lags[30:][np.argmax(cm[30:])])})
                    if f in f_profile and chem in ("succ", "acet", "mixed"):
                        for l in lags[PROFILE_LAG_MIN:]:
                            profile_rows.append({
                                "chemistry": chem, "f": f, "pH": pH, "pattern": name,
                                "lag_residues": int(l),
                                "self_mean": sm[l], "self_sd": _sd(self_r[:, l]),
                                "cross_mean": cm[l], "cross_sd": _sd(cross_r[:, l])})
                dlag_rows.append(row)
        print(f"  {chem}: done ({time.time() - t0:.0f} s)", flush=True)

    def pi_of(chem, f):
        hit = [r for r in pi_rows if r["chemistry"] == chem and np.isclose(r["f"], f)]
        return hit[0]["pI_ensemble"] if hit else float("nan")

    def interpolate_f(rows, target):
        """f at which the ensemble pI of a chemistry crosses ``target``."""
        for a, b in zip(rows, rows[1:]):
            if a["pI_ensemble"] >= target > b["pI_ensemble"]:
                return a["f"] + (b["f"] - a["f"]) * (a["pI_ensemble"] - target) / (
                    a["pI_ensemble"] - b["pI_ensemble"])
        return float("nan")

    deam = sorted((r for r in pi_rows if r["chemistry"] == "deam"), key=lambda r: r["f"])
    full_succ = modify(native, np.random.default_rng(0), f_succ=1.0)   # no randomness at f = 1
    by_set = {}
    for name in sorted(es.PKA_SETS):
        model = es.charge_model_for(name)
        by_set[name] = {
            "native_pI": es.isoelectric_point(native, ionic, model),
            "fully_succinylated_pI": es.isoelectric_point(full_succ, ionic, model),
            "native_net_charge_pH4.2": es.net_charge(native, es.Environment(4.2, ionic), model),
            "fully_succinylated_net_charge_pH4.2":
                es.net_charge(full_succ, es.Environment(4.2, ionic), model)}
    pi_native = by_set[args.pka_set]["native_pI"]
    pi_succ = by_set[args.pka_set]["fully_succinylated_pI"]
    hyd = st.axial_profile(native, native.centered_hydrophobicity())[1]
    hyd_r = np.array([st.lagged_correlation(hyd, hyd, int(l)) for l in lags])
    gates = {
        # modification leaves the hydrophobic weights unchanged, so this one
        # number holds for every ensemble
        "hydrophobicity_self_r_at_D": hyd_r[D_PERIOD_RESIDUES],
        "hydrophobicity_self_z_vs_other_lags":
            (hyd_r[D_PERIOD_RESIDUES] - hyd_r[off_d].mean()) / hyd_r[off_d].std(),
        "pka_set": args.pka_set,
        "native_pI": pi_native,
        "native_pI_near_9": bool(8.5 <= pi_native <= 9.5),
        "fully_succinylated_pI": pi_succ,
        "fully_succinylated_pI_in_4.0_4.6": bool(4.0 <= pi_succ <= 4.6),
        "by_pka_set": by_set,
        "deamidated_fraction_giving_pI_9": interpolate_f(deam, 9.0),
        "native_site_counts": {t.name: native.count(t) for t in SiteType if native.count(t)},
        "pKa": {t.name: v for t, v in es.PKA_SETS[args.pka_set].items()},
        "telopeptides": args.telopeptides, "seed": args.seed,
        "molecules_per_point": args.molecules, "f_deam_background": args.f_deam,
    }
    cc.write_csv(outdir / "pi_vs_f.csv", pi_rows)
    cc.write_csv(outdir / "net_charge.csv", charge_rows)
    cc.write_csv(outdir / "net_charge_histogram.csv", hist_rows)
    cc.write_csv(outdir / "titration.csv", titration)
    cc.write_csv(outdir / "d_lag_correlation.csv", dlag_rows)
    cc.write_csv(outdir / "autocorrelation_profile.csv", profile_rows)
    cc.write_json(outdir / "gates.json", gates)
    print(f"native pI {pi_native:.2f} (gate 'near 9': "
          f"{'pass' if gates['native_pI_near_9'] else 'FAIL'}); fully succinylated pI "
          f"{pi_succ:.2f} (gate 4.0-4.6: "
          f"{'pass' if gates['fully_succinylated_pI_in_4.0_4.6'] else 'FAIL'})")
    if not args.no_figures:
        figure_pi(outdir / "fig_pi_vs_f.png", pi_rows, args.target_pH)
        figure_net_charge(outdir / "fig_net_charge.png", titration, charge_rows,
                          f_profile, args.ph)
        figure_charge_distribution(outdir / "fig_net_charge_distribution.png", hist_rows,
                                   [f for f in (0.5, 0.8, 0.9, 0.97) if f in args.f],
                                   args.target_pH)
        for pH in corr_ph:
            figure_autocorrelation(outdir / f"fig_charge_autocorrelation_pH{pH:g}.png",
                                   profile_rows, dlag_rows, pH)
    print(f"wrote {outdir}/  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
