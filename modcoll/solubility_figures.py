"""Figures of the pair-interaction solubility measure (``modcoll.cli_solubility``).

Sign convention in every map: red = B2 > 0 (net repulsion: the solution is
stable), blue = B2 < 0 (net attraction: aggregation). The colour value is
sign(B2*) log10(1 + |B2*|), B2* = B2 / B2_hc, so hard rods (B2* = 1) are
pale red and the black line is B2 = 0.
"""
from __future__ import annotations

import numpy as np

from . import cli_common as cc
from . import composition as co
from .composition_figures import DIVERGING, _cmap, _triangle, ternary_xy

TERM_LABELS = {"full": "full energy", "el": "charges only", "hy": "hydrophobic only",
               "net_hy": "net charge (no pattern)\n+ hydrophobic", "net": "net charge only"}


def signed_log(b2_star):
    b = np.asarray(b2_star, dtype=float)
    return np.sign(b) * np.log10(1.0 + np.abs(b))


def _grid(rows, term, temperature, key="B2_star"):
    sel = [r for r in rows if r["term"] == term and r["temperature_C"] == temperature]
    ph = sorted({r["pH"] for r in sel})
    ionic = sorted({r["ionic_strength_M"] for r in sel})
    z = np.full((len(ionic), len(ph)), np.nan)
    for r in sel:
        z[ionic.index(r["ionic_strength_M"]), ph.index(r["pH"])] = r[key]
    return np.array(ph), 1e3 * np.array(ionic), z


def _map(ax, ph, mM, b2_star, vmax=6.0, contour=True):
    from matplotlib.colors import TwoSlopeNorm
    v = np.clip(signed_log(b2_star), -vmax, vmax)
    m = ax.pcolormesh(ph, mM, v, cmap=_cmap(DIVERGING, "b2"), shading="nearest",
                      norm=TwoSlopeNorm(0.0, -vmax, vmax))
    if contour and np.nanmin(v) < 0 < np.nanmax(v) and len(ph) > 1 and len(mM) > 1:
        ax.contour(ph, mM, v, levels=[0.0], colors=[cc.INK], linewidths=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("pH")
    ax.set_ylabel("ionic strength (mM)")
    ax.grid(False)
    return m


def figure_native(path, rows, key_rows, d_nm):
    plt = cc.pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    ph, mM, z = _grid(rows, "full", 25.0)
    m = _map(axes[0], ph, mM, z)
    for label, pH, i, mk in (("acid stock", 3.0, 10.0, "o"), ("physiological", 7.4, 150.0, "s")):
        axes[0].plot(pH, i, mk, mfc="none", mec=cc.INK, ms=8, mew=1.2)
        axes[0].annotate(label, (pH, i), xytext=(6, 6), textcoords="offset points", fontsize=7.5)
    axes[0].set_title("Native collagen, 25 C: sign of B2")
    bar = fig.colorbar(m, ax=axes[0], fraction=0.05, pad=0.02)
    bar.set_label("sign(B2*) log10(1 + |B2*|)", fontsize=8)
    bar.outline.set_visible(False)
    # B2 = 0 lines at the three temperatures
    ax = axes[1]
    for (t, color) in zip((4.0, 25.0, 37.0), cc.CATEGORICAL[:3]):
        ph, mM, z = _grid(rows, "full", t)
        if np.nanmin(z) < 0 < np.nanmax(z):
            ax.contour(ph, mM, signed_log(z), levels=[0.0], colors=[color], linewidths=1.6)
        ax.plot([], [], color=color, label=f"{t:g} C")
    ax.set_yscale("log")
    ax.set_xlabel("pH")
    ax.set_ylabel("ionic strength (mM)")
    ax.set_title("Where B2 changes sign (attractive inside)")
    ax.legend(loc="upper left", fontsize=7.5)
    # decomposition at the two reference conditions
    ax = axes[2]
    picks = [r for r in key_rows if r["condition"] in
             ("dilute acid, low salt", "neutral, physiological, warm")]
    terms = list(TERM_LABELS)
    width = 0.38
    for k, (r, color) in enumerate(zip(picks, (cc.CATEGORICAL[1], cc.CATEGORICAL[0]))):
        vals = [r["B2_star"] if t == "full" else r[f"B2_star_{t}"] for t in terms]
        x = np.arange(len(terms)) + (k - 0.5) * width
        ax.bar(x, signed_log(vals), width, color=color,
               label=f"pH {r['pH']:g}, {r['ionic_mM']:g} mM, {r['temperature_C']:g} C")
    ax.axhline(0.0, color=cc.INK, lw=0.8)
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels([TERM_LABELS[t] for t in terms], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("sign(B2*) log10(1 + |B2*|)")
    ax.set_title("Which part of the energy decides")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 0.0) + 0.45 * (max(hi, 0.0) - lo))
    ax.legend(fontsize=7.5, loc="upper right")
    rect = cc.headline(fig, "Native control: soluble in dilute acid, attractive at neutral pH",
                       "aligned-rod second virial coefficient; red = net repulsion (soluble), "
                       "blue = net attraction; B2* = B2 / hard-rod value")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_native_salt(path, rows):
    plt = cc.pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    styles = cc.level_styles(3)
    for ax, key, label in ((axes[0], "B2_star", "sign(B2*) log10(1 + |B2*|)"),
                           (axes[1], "F_best_kT", "deepest bound state F_best (kT)")):
        for pH, ls in ((3.0, "-"), (7.0, "--")):
            for (t, (color, mk)) in zip((4.0, 25.0, 37.0), styles):
                sel = sorted((r for r in rows if r["term"] == "full" and r["pH"] == pH
                              and r["temperature_C"] == t), key=lambda r: r["ionic_strength_M"])
                x = [1e3 * r["ionic_strength_M"] for r in sel]
                y = [r[key] for r in sel]
                if key == "B2_star":
                    y = signed_log(y)
                ax.plot(x, y, ls, color=color, marker=mk, ms=3.5,
                        label=f"pH {pH:g}, {t:g} C")
        ax.set_xscale("log")
        ax.set_xlabel("ionic strength (mM)")
        ax.set_ylabel(label)
        ax.axhline(0.0, color=cc.INK, lw=0.8)
    axes[0].set_title("Salt: screening the repulsion of the acid form")
    axes[1].set_title("Depth of the best bound state")
    axes[0].legend(fontsize=7, ncol=2)
    rect = cc.headline(fig, "Native collagen against ionic strength",
                       "solid: pH 3 (acid form); dashed: pH 7; colours: temperature")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_modified(path, rows, bench):
    plt = cc.pyplot()
    comps = list(dict.fromkeys(r["composition"] for r in rows))
    n = len(comps)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.0, 2.75 * nrow + 0.6), squeeze=False)
    m = None
    for k, name in enumerate(comps):
        ax = axes[k // ncol, k % ncol]
        sel = [r for r in rows if r["composition"] == name]
        ph, mM, z = _grid(sel, "full", 25.0)
        m = _map(ax, ph, mM, z)
        for _, pH, i, expect in bench:
            ax.plot(pH, i, "o" if expect == "soluble" else "X", mfc="none" if
                    expect == "soluble" else cc.INK, mec=cc.INK, ms=6, mew=1.0)
        ax.set_title(name, fontsize=8.5)
        if k % ncol:
            ax.set_ylabel("")
        if k // ncol < nrow - 1:
            ax.set_xlabel("")
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    rect = cc.headline(fig, "Modified collagen: where the pair attracts (blue) or repels "
                       "(red), 25 C",
                       "circles: bench conditions where the material dissolves; crosses: "
                       "where it precipitates; mean B2 over random pattern pairs")
    fig.tight_layout(rect=rect)
    if m is not None:
        bar = fig.colorbar(m, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        bar.set_label("sign(B2*) log10(1 + |B2*|)", fontsize=8)
        bar.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def figure_inverse(path, rows, target):
    plt = cc.pyplot()
    names = list(dict.fromkeys(r["variant"] for r in rows))
    levels = sorted({r["f_deam"] for r in rows})
    fig, axes = plt.subplots(len(names), len(levels),
                             figsize=(2.2 * len(levels), 2.45 * len(names) + 0.9), squeeze=False)
    for i, name in enumerate(names):
        sub = [r for r in rows if r["variant"] == name]
        pts = np.array([[r["f_succ"], r["f_acet"], r["f_deam"]] for r in sub])
        pI = np.array([r["pI_ensemble"] for r in sub])
        ok = np.array([r["matches_all_pair_criteria"] for r in sub])
        band = co.in_band(pI, target, 0.3)
        cats = (("pI in 4.2 +- 0.3, pair criteria met", band & ok, cc.CATEGORICAL[0], "o"),
                ("pI in band, pair criteria not met", band & ~ok, cc.CATEGORICAL[1], "X"),
                ("pI outside band, pair criteria met", ~band & ok, cc.INK_SECONDARY, "o"),
                ("pI outside band, criteria not met", ~band & ~ok, cc.GRID, "X"))
        for j, (ax, fd) in enumerate(zip(axes[i], levels)):
            _triangle(ax, corner_labels=(i == len(names) - 1 and j == 0))
            for label, mask, color, mk in cats:
                sel = mask & np.isclose(pts[:, 2], fd)
                x, y = ternary_xy(pts[sel, 0], pts[sel, 1])
                ax.scatter(x, y, s=30, marker=mk, color=color, label=label, zorder=5,
                           edgecolors="none")
            if i == 0:
                ax.set_title(f"deamidated {fd:g}" + (" (liming)" if np.isclose(
                    fd, co.LIMING_F_DEAM) else ""), fontsize=8.5)
            if j == 0:
                ax.text(-0.05, 0.5, name, transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5)
    rect = cc.headline(fig, "Inverse problem with the pair criterion",
                       "rows: parameter sets; criteria: B2 > 0 in deionized water, at pH 7 and "
                       "in PBS, B2 < 0 in acetate at pH 4.2")
    fig.tight_layout(rect=(rect[0], 0.6 / fig.get_figheight(), rect[2], rect[3]))
    fig.savefig(path)
    plt.close(fig)


def figure_water(path, rows, fits):
    plt = cc.pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    comps = [f["composition"] for f in fits]
    show = [c for c in comps if c in ("native", "succinylated 90 %", "c0 most succinylation",
                                      "c1 least succinylation", "c3 most deamidation")]
    styles = cc.level_styles(len(show))
    meas = [r for r in rows if r["composition"] == show[0]]
    x = [r["conc_g_per_L"] / 10.0 for r in meas]
    axes[0].plot(x, [r["pH_measured"] for r in meas], "o", color=cc.INK, ms=6,
                 label="measured", zorder=6)
    for name, (color, mk) in zip(show, styles):
        sel = [r for r in rows if r["composition"] == name]
        axes[0].plot(x, [r["pH_predicted_fixed_acid"] for r in sel], "-", color=color,
                     marker=mk, ms=3.5, label=f"{name} (fit)")
        axes[1].plot(x, [r["counter_anions_per_molecule"] for r in sel], "-", color=color,
                     marker=mk, ms=3.5, label=name)
        if name != "native":
            axes[0].axhline(sel[0]["pI"], color=color, lw=0.7, ls=":")
    ticks = [r["conc_g_per_L"] / 10.0 for r in meas]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.3g}" for t in ticks], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlabel("concentration (% w/v)")
    axes[0].set_ylabel("pH")
    axes[0].set_title("pH of the material in water: one fixed counter-ion content")
    axes[1].set_ylabel("strong-acid anions per molecule")
    axes[1].set_title("Counter-ions each reading implies")
    axes[1].axhline(0.0, color=cc.INK, lw=0.8)
    axes[0].legend(fontsize=7)
    rect = cc.headline(fig, "The material in deionized water",
                       "dotted: isoelectric point of each composition; the last reading "
                       "cannot be reached with the counter-ion content of the others")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def write_all(outdir, results, d_nm):
    from .cli_solubility import BENCH
    if "native_rows" in results:
        figure_native(outdir / "fig_native_control.png", results["native_rows"],
                      results["native_key"], d_nm)
        figure_native_salt(outdir / "fig_native_salt.png", results["native_rows"])
    if "modified_rows" in results:
        figure_modified(outdir / "fig_modified_maps.png", results["modified_rows"], BENCH)
    if "inverse_rows" in results:
        figure_inverse(outdir / "fig_inverse.png", results["inverse_rows"], 4.2)
    if "water_rows" in results:
        figure_water(outdir / "fig_water.png", results["water_rows"], results["water_fits"])
