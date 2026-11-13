"""Figures of the composition scan (``modcoll.cli_composition``).

The amine composition (unmodified, succinylated, acetylated; the three sum to
one) is drawn as a ternary diagram, one diagram per deamidated fraction.
Colours follow ``cli_common``: one blue ramp for ordered quantities, a
blue-grey-red pair for signed charge, the categorical slots for classes.
"""
from __future__ import annotations

import numpy as np

from . import cli_common as cc
from . import composition as co

# One-hue ramp, light to dark, for magnitudes and ordered levels.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
# Signed charge: negative blue, positive red, neutral grey at zero.
DIVERGING = ("#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#a82a2a")
BAND_COLORS = ("#1c5cab", "#5598e7", "#b7d3f6")     # narrowest band darkest
SLICES = (0.0, co.LIMING_F_DEAM, 0.5, 1.0)
REGION_SLICES = (0.0, co.LIMING_F_DEAM, 0.4, 0.6, 0.8, 1.0)
_H = np.sqrt(3.0) / 2.0


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(colors))


def ternary_xy(f_succ, f_acet):
    """Cartesian position in the amine triangle: unmodified at the lower
    left, fully acetylated at the lower right, fully succinylated at the top."""
    f_succ = np.asarray(f_succ, dtype=float)
    return np.asarray(f_acet, dtype=float) + 0.5 * f_succ, _H * f_succ


def _triangle(ax, corner_labels=True, ticks=True):
    ax.plot([0, 1, 0.5, 0], [0, 0, _H, 0], color=cc.INK_SECONDARY, lw=0.8, zorder=4)
    if ticks:
        for f in (0.25, 0.5, 0.75):
            for (a, b) in (((f, 0.0), (0.0, f)), ((f, 0.0), (f, 1.0 - f)),
                           ((0.0, f), (1.0 - f, f))):
                # a, b are (f_succ, f_acet) end points of a grid line
                x, y = ternary_xy([a[0], b[0]], [a[1], b[1]])
                ax.plot(x, y, color=cc.SURFACE, lw=0.5, alpha=0.6, zorder=3)
    if corner_labels:
        kw = dict(fontsize=7.5, color=cc.INK_SECONDARY)
        ax.text(0.0, -0.035, "unmodified", ha="left", va="top", **kw)
        ax.text(1.0, -0.035, "all acetylated", ha="right", va="top", **kw)
        ax.text(0.47, _H, "all succinylated", ha="right", va="center", **kw)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.11, _H + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")


def _slice(points, f_deam):
    return np.flatnonzero(np.isclose(points[:, 2], f_deam))


def _levels(points, wanted):
    """The deamidation levels of the grid nearest the wanted ones, once each."""
    have = np.unique(points[:, 2])
    out = []
    for w in wanted:
        level = float(have[np.argmin(np.abs(have - w))])
        if level not in out:
            out.append(level)
    return out


def _slice_title(f_deam):
    if np.isclose(f_deam, co.LIMING_F_DEAM):
        return f"deamidated {f_deam:g} (liming level)"
    return f"deamidated {f_deam:g}"


def figure_forward_map(path, points, stats, target):
    from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
    from matplotlib.tri import Triangulation
    plt = cc.pyplot()
    tag = f"pH{target:g}"
    pI = np.array([s["pI_ensemble"] for s in stats])
    q_t = np.array([s[f"net_charge_{tag}_mean"] for s in stats])
    q_7 = np.array([s["net_charge_pH7_mean"] for s in stats])
    pi_levels = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.75, 5.0, 6.0, 8.0, 10.0]
    pi_cmap = _cmap(BLUE_RAMP[::-1][:len(pi_levels) + 1], "pi")
    div = _cmap(DIVERGING, "charge")
    rows = (
        ("isoelectric point of the sample", pI, dict(levels=pi_levels, cmap=pi_cmap,
                                                     norm=BoundaryNorm(pi_levels, pi_cmap.N)),
         "pH"),
        (f"net charge at pH {target:g}", q_t,
         dict(levels=[-160, -120, -80, -40, -20, -5, 5, 20, 40, 80, 120, 160], cmap=div,
              norm=TwoSlopeNorm(0.0, -160, 160)), "e per molecule"),
        ("net charge at pH 7", q_7,
         dict(levels=[-320, -240, -160, -120, -80, -40, -10, 10, 40, 80, 120, 160, 240, 320],
              cmap=div, norm=TwoSlopeNorm(0.0, -320, 320)), "e per molecule"),
    )
    slices = _levels(points, SLICES)
    fig, axes = plt.subplots(len(rows), len(slices) + 1, figsize=(11.4, 8.3), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(slices) + [0.06]})
    for r, (label, values, kw, unit) in enumerate(rows):
        mappable = None
        for c, f_deam in enumerate(slices):
            ax = axes[r, c]
            sel = _slice(points, f_deam)
            x, y = ternary_xy(points[sel, 0], points[sel, 1])
            v = np.clip(values[sel], kw["levels"][0], kw["levels"][-1])
            mappable = ax.tricontourf(Triangulation(x, y), v, **kw)
            if r == 0:
                ax.tricontour(Triangulation(x, y), values[sel], levels=[target],
                              colors=[cc.INK], linewidths=0.9)
                ax.text(0.5, 1.2, _slice_title(f_deam), transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=8.5, color=cc.INK_SECONDARY)
            _triangle(ax, corner_labels=(c == 0))
            if c == 0:
                ax.text(0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9, fontweight="bold")
        bar = fig.colorbar(mappable, cax=axes[r, -1])
        bar.set_label(unit, fontsize=8)
        bar.ax.tick_params(labelsize=7)
        bar.outline.set_visible(False)
    rect = cc.headline(fig, "Forward map: what each composition gives",
                       "one triangle = all splits of the amines; columns = deamidated "
                       f"fraction of Asn/Gln; black line = isoelectric point {target:g}")
    fig.tight_layout(rect=rect, h_pad=1.6)
    fig.savefig(path)
    plt.close(fig)


def figure_region(path, points, pI, target, bands, rep_rows):
    from matplotlib.patches import Patch
    from matplotlib.tri import Triangulation
    plt = cc.pyplot()
    bands = sorted(bands)
    levels = [target - b for b in bands[::-1]] + [target + b for b in bands]
    colors = list(BAND_COLORS[:len(bands)][::-1]) + list(BAND_COLORS[:len(bands)])[1:]
    slices = _levels(points, REGION_SLICES)
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.2), squeeze=False)
    for ax in axes.ravel()[len(slices):]:
        ax.axis("off")
    for ax, f_deam in zip(axes.ravel(), slices):
        sel = _slice(points, f_deam)
        x, y = ternary_xy(points[sel, 0], points[sel, 1])
        if np.nanmin(pI[sel]) <= levels[-1]:
            ax.tricontourf(Triangulation(x, y), pI[sel], levels=levels, colors=colors)
        _triangle(ax)
        ax.set_title(_slice_title(f_deam), fontsize=9)
        for r in rep_rows:
            if np.isclose(r["f_deam"], f_deam):
                px, py = ternary_xy(r["f_succ"], r["f_acet"])
                ax.plot(px, py, "o", ms=5, color=cc.INK, mec=cc.SURFACE, mew=1.2, zorder=6)
                ax.annotate(f"c{r['composition']}", (px, py), textcoords="offset points",
                            xytext=(6, 2), fontsize=7.5, color=cc.INK, zorder=6)
    handles = [Patch(color=BAND_COLORS[k], label=f"isoelectric point {target:g} ± {b:g}")
               for k, b in enumerate(bands)]
    fig.legend(handles=handles, loc="lower center", ncols=len(bands), fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005))
    rect = cc.headline(fig, "Compositions consistent with the observed isoelectric point",
                       "blank = outside every band; dots = compositions taken to the "
                       "stagger scan")
    fig.tight_layout(rect=(rect[0], 0.05, rect[2], rect[3]))
    fig.savefig(path)
    plt.close(fig)


def figure_min_succinylation(path, min_succ, sens_curves, bands, n_variants):
    plt = cc.pyplot()
    bands = sorted(bands)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.5 * len(bands) + 0.6, 3.9),
                             sharey=True, squeeze=False)
    for ax, band in zip(axes[0], bands):
        rows = sorted((r for r in sens_curves if np.isclose(r["band"], band)),
                      key=lambda r: r["f_deam_cap"])
        if rows:
            x = np.array([r["f_deam_cap"] for r in rows])
            ax.fill_between(x, [r["f_succ_min_lowest_variant"] for r in rows],
                            [r["f_succ_min_highest_variant"] for r in rows],
                            color=BLUE_RAMP[1], lw=0,
                            label=f"range over {n_variants} parameter sets")
            ax.plot(x, [r["f_succ_min_base"] for r in rows], color=cc.CATEGORICAL[0], lw=2,
                    label="base parameters")
            missing = max(r["variants_without_a_solution"] for r in rows)
            if missing:
                first = max((r["f_deam_cap"] for r in rows
                             if r["variants_without_a_solution"]), default=0.0)
                ax.annotate(f"up to {missing} parameter sets have\nno solution at or below "
                            f"{first:g}", (0.98, 0.97), xycoords="axes fraction", ha="right",
                            va="top", fontsize=7.5, color=cc.INK_SECONDARY)
        else:
            rows = sorted((r for r in min_succ if np.isclose(r["band"], band) and r["reached"]),
                          key=lambda r: r["f_deam_cap"])
            ax.plot([r["f_deam_cap"] for r in rows], [r["f_succ_min"] for r in rows],
                    color=cc.CATEGORICAL[0], lw=2, marker="o", label="base parameters")
        ax.axvline(co.LIMING_F_DEAM, color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)))
        ax.annotate("liming", (co.LIMING_F_DEAM, 1.0), textcoords="offset points",
                    xytext=(3, -9), fontsize=7.5, color=cc.INK_SECONDARY)
        ax.set_title(f"isoelectric point within ± {band:g}")
        ax.set_xlabel("deamidated fraction allowed (at most)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("least succinylated fraction\nof any consistent composition")
    axes[0, -1].legend(loc="center right", fontsize=8)
    rect = cc.headline(fig, "How much succinylation a consistent composition needs",
                       "the rest of the amines may be acetylated; less is needed the more "
                       "deamidation is allowed")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_sensitivity(path, sens_rows, band):
    plt = cc.pyplot()
    params = (("succinyl_pKa", "succinyl pKa"), ("asp_pKa", "Asp pKa"), ("glu_pKa", "Glu pKa"),
              ("chain_ends", "chain ends"), ("species", "sequence"))
    names = {"mature": "intact", "pepsin": "pepsin-cut", "human": "human",
             "bovine_porcine": "bovine / porcine"}
    quantities = (("pure_succinylation_lowest_pI", "isoelectric point at\ncomplete succinylation"),
                  ("pure_succinylation_f_reaching_band",
                   f"succinylated fraction at which\nsuccinylation alone enters ± {band:g}"),
                  ("f_succ_min_liming_or_less",
                   f"least succinylation with deamidation\nat or below liming (± {band:g})"))
    rows = [r for r in sens_rows if np.isclose(r["band"], band)]
    base = next(r for r in rows if r["is_base"])
    base_keys = {k: base[k] for k, _ in params}
    fig, axes = plt.subplots(1, len(quantities), figsize=(11.2, 3.9), sharey=True,
                             squeeze=False)
    for ax, (key, label) in zip(axes[0], quantities):
        for y, (param, _) in enumerate(params):
            alt = [r for r in rows if all(r[k] == v for k, v in base_keys.items() if k != param)]
            alt.sort(key=lambda r: str(r[param]))
            xs = np.array([r[key] for r in alt], dtype=float)
            ok = np.isfinite(xs)
            if ok.any():
                ax.plot([xs[ok].min(), xs[ok].max()], [y, y], color=BLUE_RAMP[3], lw=2,
                        solid_capstyle="round", zorder=2)
            for r, x in zip(alt, xs):
                if not np.isfinite(x):
                    continue
                is_base = r[param] == base_keys[param]
                ax.plot(x, y, "o", ms=6, color=cc.INK if is_base else cc.CATEGORICAL[0],
                        mec=cc.SURFACE, mew=1.0, zorder=3)
                if not is_base:
                    text = names.get(r[param], f"{r[param]:g}" if not isinstance(
                        r[param], str) else r[param])
                    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7.5, color=cc.INK_SECONDARY)
        if np.isfinite(base[key]):
            ax.axvline(base[key], color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)), zorder=1)
        ax.set_xlabel(label)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([p[1] for p in params])
        ax.set_ylim(len(params) - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
    rect = cc.headline(fig, "One parameter changed at a time",
                       "black dot and dashed line = base parameters; blue dots = the other "
                       "values of that parameter, labelled")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_assays(path, ranking_rows, assay_rows, band):
    plt = cc.pyplot()
    rank = sorted((r for r in ranking_rows if np.isclose(r["band"], band)),
                  key=lambda r: r["rank"])
    cand = [r for r in assay_rows if r["band"] <= band + 1e-9 and not r["liming_reference"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(rank))
    ax.barh(y, [r["pairs_separated"] for r in rank], height=0.55, color=cc.CATEGORICAL[0],
            zorder=2)
    for k, r in enumerate(rank):
        lo, hi = r["pairs_separated_sigma_x2"], r["pairs_separated_sigma_x0.5"]
        ax.plot([lo, hi], [k + 0.38, k + 0.38], color=cc.INK_SECONDARY, lw=1.0, zorder=3)
        ax.annotate(f"{100 * r['pairs_separated']:.0f} %", (max(r["pairs_separated"], hi), k),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rank])
    ax.set_ylim(len(rank) - 0.3, -0.7)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("share of candidate pairs told apart at 2 sd\n"
                  "(thin line: assay precision doubled ... halved)")
    ax.grid(axis="y", visible=False)
    ax.set_title("each assay alone")

    cmap = _cmap(BLUE_RAMP[3:], "succ")
    sc = ax2.scatter([r["free_amine_fraction"] for r in cand],
                     [1e-3 * r["mass_shift_Da"] for r in cand],
                     c=[r["f_succ"] for r in cand], cmap=cmap, vmin=0, vmax=1, s=9, lw=0)
    bar = fig.colorbar(sc, ax=ax2, pad=0.02)
    bar.set_label("succinylated fraction", fontsize=8)
    bar.outline.set_visible(False)
    ax2.set_xlabel("free amines remaining (fraction of native)")
    ax2.set_ylabel("mass shift (kDa per molecule)")
    ax2.set_title("free amines and mass together")
    rect = cc.headline(fig, "Which bench assay identifies the composition",
                       f"candidates = compositions with isoelectric point within ± {band:g} "
                       f"({len(cand)} on the grid)")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_titration(path, curve_rows, target):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    plt = cc.pyplot()
    comps = sorted({r["composition"] for r in curve_rows})
    cmap = _cmap(BLUE_RAMP[3:], "succ")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 0.04]})
    for ax, (xlim, ylim) in zip(axes[:2], (((2.0, 12.0), None), ((3.4, 5.2), (-130, 130)))):
        for c in comps:
            sel = sorted((r for r in curve_rows if r["composition"] == c),
                         key=lambda r: r["pH"])
            x = [r["pH"] for r in sel]
            y = [r["net_charge_mean"] for r in sel]
            if c < 0:
                ax.plot(x, y, color=cc.INK, lw=1.2, ls="--", label="unmodified")
            else:
                ax.plot(x, y, color=cmap(sel[0]["f_succ"]), lw=1.6)
        ax.axhline(0.0, color=cc.INK_SECONDARY, lw=0.8)
        ax.axvline(target, color=cc.INK_SECONDARY, lw=0.8, ls=(0, (2, 3)))
        ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("pH")
    axes[0].set_ylabel("mean net charge per molecule (e)")
    axes[0].set_title("whole titration")
    axes[1].set_title(f"around pH {target:g}")
    axes[0].legend(loc="upper right", fontsize=8)
    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), cax=axes[2])
    bar.set_label("succinylated fraction", fontsize=8)
    bar.outline.set_visible(False)
    rect = cc.headline(fig, "Net charge against pH for compositions that share the "
                       "isoelectric point", "one curve per composition taken to the stagger "
                       "scan; they cross zero together and fan out above pH 5")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def _composition_label(r):
    return f"c{r['composition']}   {r['f_succ']:.2f} / {r['f_acet']:.2f} / {r['f_deam']:.2f}"


def figure_stagger(path, stag_summary, rep_rows):
    plt = cc.pyplot()
    par = [r for r in stag_summary if r["orientation"] == "parallel"]
    phs = sorted({r["pH"] for r in par})
    salts = sorted({r["ionic_mM"] for r in par})
    cmap = _cmap(BLUE_RAMP, "share")
    fig, axes = plt.subplots(1, len(phs) + 1, figsize=(2.15 * len(phs) + 2.6,
                                                       0.42 * len(rep_rows) + 2.1),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(phs) + [0.05]})
    image = None
    for ax, pH in zip(axes[0], phs):
        grid = np.full((len(rep_rows), len(salts)), np.nan)
        for i, rep in enumerate(rep_rows):
            for j, salt in enumerate(salts):
                r = next(x for x in par if x["composition"] == rep["composition"]
                         and np.isclose(x["pH"], pH) and np.isclose(x["ionic_mM"], salt))
                grid[i, j] = r["pairs_deepest_at_multiple_mean"]
                ax.text(j, i, f"{r['deepest_D']:.2f}", ha="center", va="center", fontsize=7,
                        color=cc.SURFACE if grid[i, j] > 0.55 else cc.INK)
        image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(salts)))
        ax.set_xticklabels([f"{s:g}" for s in salts])
        ax.set_yticks(range(len(rep_rows)))
        ax.set_yticklabels([_composition_label(r) for r in rep_rows] if ax is axes[0, 0]
                           else [])
        ax.set_xticks(np.arange(len(salts) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rep_rows) + 1) - 0.5, minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=cc.SURFACE, lw=2)
        ax.tick_params(which="both", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"pH {pH:g}")
        ax.set_xlabel("ionic strength (mM)")
    bar = fig.colorbar(image, cax=axes[0, -1])
    bar.set_label("share of pattern pairs whose deepest\nminimum is at a D multiple",
                  fontsize=8)
    bar.outline.set_visible(False)
    rect = cc.headline(fig, "Does the D register depend on the composition?",
                       "rows = succinylated / acetylated / deamidated; "
                       "cell number = deepest minimum (D)")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def figure_crossover(path, crossovers):
    plt = cc.pyplot()
    classes = (("succinyl-rich (0.75 or more)", lambda f: f >= 0.75),
               ("intermediate", lambda f: 0.25 < f < 0.75),
               ("little succinyl (0.25 or less)", lambda f: f <= 0.25))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    salts = [r["crossover_mM"] for r in crossovers if np.isfinite(r["crossover_mM"])]
    for k, (label, member) in enumerate(classes):
        rows = [r for r in crossovers if member(r["f_succ"])]
        inside = [r for r in rows if r["status"] == "bracketed"]
        below = [r for r in rows if r["status"] == "below_range"]
        ax.plot([r["net_charge_mean"] for r in inside], [r["crossover_mM"] for r in inside],
                cc.MARKERS[k], ms=7, color=cc.CATEGORICAL[k], mec=cc.SURFACE, mew=1.0,
                ls="none", label=label)
        ax.plot([r["net_charge_mean"] for r in below], [r["crossover_mM"] for r in below],
                cc.MARKERS[k], ms=7, mfc="none", mec=cc.CATEGORICAL[k], mew=1.3, ls="none")
    ax.axvline(0.0, color=cc.INK_SECONDARY, lw=0.8)
    ax.set_yscale("log")
    if salts:
        ticks = [10, 20, 50, 100, 150]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel("mean net charge of a molecule at that pH (e)")
    ax.set_ylabel("ionic strength at which half the pairs\nprefer a D multiple (mM)")
    ax.legend(loc="upper right", fontsize=8, title="succinylated fraction", title_fontsize=8)
    rect = cc.headline(fig, "Salt needed to keep the D register",
                       "one point per composition and pH; open marker = already above one "
                       "half at the lowest salt scanned")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def write_all(outdir, data: dict) -> None:
    """Write every figure the stages that ran have data for."""
    d = data
    target, bands = d["target"], sorted(d["bands"])
    middle = bands[len(bands) // 2]
    figure_forward_map(outdir / "fig_forward_map.png", d["points"], d["stats"], target)
    if d["masks"]:
        figure_region(outdir / "fig_consistent_region.png", d["points"], d["pI"], target,
                      bands, d["rep_rows"])
        figure_min_succinylation(outdir / "fig_min_succinylation.png", d["min_succ"],
                                 d["sens_curves"], bands, d["n_variants"])
    if d["sens_rows"]:
        figure_sensitivity(outdir / "fig_sensitivity.png", d["sens_rows"], middle)
    if d["ranking_rows"]:
        figure_assays(outdir / "fig_assays.png", d["ranking_rows"], d["assay_rows"], middle)
    if d["curve_rows"]:
        figure_titration(outdir / "fig_titration.png", d["curve_rows"], target)
    if d["stag_summary"]:
        figure_stagger(outdir / "fig_stagger_robustness.png", d["stag_summary"], d["rep_rows"])
        figure_crossover(outdir / "fig_stagger_crossover.png", d["crossovers"])
