"""S3: single ion-site validation and the two-row bridging demonstration.

    python -m modcoll.ions.validate_s3            # full run, about 15 minutes on 8 cores
    python -m modcoll.ions.validate_s3 --quick    # smoke run, about one minute

Writes to results/modcoll/ions/:

    s3_binding_table.json / .md   association constants, free energies and
                                  contact geometry per ion-ligand pair, against
                                  the targets of params.toml
    s3_binding_constants.png      model against target, and the uncalibrated
                                  ionic-strength check
    s3_inner_outer.png            inner- and outer-sphere shares, Ca2+ vs Mg2+
    s3_distances.png              ion-ligand distance distributions
    s3_bridges_vs_gap.png         bridge probability vs gap and bath
                                  concentration for two facing rows
    s3_bridge_kinds.png           the same split into inner-inner, inner-outer
                                  and outer-outer bridges
    s3_bridges.json               the numbers behind the two bridge figures

Every random number comes from numpy Generators seeded from --seed.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import calibrate as cal
from . import geometry as geo
from .bridges import INNER, BridgeAnalyzer, BridgeTracker
from .params import IonParams, load_params
from .sampler import IonSampler

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "modcoll" / "ions"

# Row demonstration geometry
ROW_SPACING = 0.86      # nm between neighbouring groups along a row: three
                        # residues at the collagen rise of 0.286 nm per residue
ROW_GROUPS = 8          # groups per row; Lz = 8 * 0.86 = 6.88 nm, periodic
CORE_RADIUS = 0.5       # nm, excluded cylinder behind each row
BOX_XY = 6.0            # nm
ROW_IONIC_STRENGTH = 0.1  # mol/L monovalent background (acetate buffer + salt)
ROW_PH = 4.2            # the carboxyl pKa is constants.carboxyl_pKa of params.toml

# Colours: fixed per entity (validated categorical slots 1-3), text in ink
C_ION = {"Ca": "#2a78d6", "Mg": "#eb6834", "SO4": "#1baf7a"}
C_KIND = {"inner-inner": "#2a78d6", "inner-outer": "#eb6834", "outer-outer": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ION_LABEL = {"Ca": "Ca$^{2+}$", "Mg": "Mg$^{2+}$", "SO4": "SO$_4^{2-}$"}


# --------------------------------------------------------------------------
# Part 1: single ion, single ligand
# --------------------------------------------------------------------------
def binding_table(params: IonParams, seed: int, n_samples: int) -> list[dict]:
    rows = cal.report(params, seed, n_samples)
    for row, spec in zip(rows, cal.FITS):
        a01 = cal.evaluate(params, spec, seed, n_samples, ionic_strength=0.1)
        row["K_total_at_I_0.1"] = a01.K_total
        row["species"] = spec.species
        row["ligand_type"] = spec.ligand_type
    return rows


def counterfactual(params: IonParams, seed: int, n_samples: int) -> dict:
    """Inner-sphere share of Mg-acetate if Mg2+ gained only as much on contact
    as Ca2+ does while keeping its own desolvation cost: what the hydration
    shell argument alone would predict."""
    p = params.with_overrides({
        "contact.Mg.O_carboxylate.eps_in": float(params.value("contact.Ca.O_carboxylate.eps_in"))})
    a = cal.evaluate(p, cal.FITS[1], seed, n_samples)
    return {"K_inner": a.K_inner, "K_total": a.K_total, "inner_fraction": a.inner_fraction}


def single_pair_run(args):
    """Canonical run of one ion next to one ligand group: contact classes and
    ion-partner distances. Returns histograms and class populations."""
    species, ligand_name, seed, n_samples, stride = args
    params = load_params()
    builders = {"acetate": cal.acetate, "methylguanidinium": cal.methylguanidinium}
    box = np.array([3.2, 3.2, 3.2])
    lig = builders[ligand_name]().translated(box / 2)
    I = 0.1
    s = IonSampler(params, lig, box, [True] * 3, I, np.random.default_rng(seed))
    s.add_ion(species, [0.4, 0.4, 0.4])
    an = BridgeAnalyzer(params, lig, box, [True] * 3)
    s.sweep(20000)
    edges = np.linspace(0.15, 0.80, 131)
    hist = {"inner": np.zeros(len(edges) - 1), "outer": np.zeros(len(edges) - 1)}
    pop = {"unbound": 0, "outer": 0, "inner_mono": 0, "inner_multi": 0}
    inner_d, bite, off_axis = [], [], []
    for _ in range(n_samples):
        s.sweep(stride)
        snap = s.snapshot()
        c = an.contacts(snap)
        if len(c) == 0:
            pop["unbound"] += 1
            continue
        n_in = int((c.cls == INNER).sum())
        if n_in == 0:
            pop["outer"] += 1
            hist["outer"] += np.histogram(c.dist.min(), bins=edges)[0]
        else:
            pop["inner_mono" if n_in == 1 else "inner_multi"] += 1
            d_in = c.dist[c.cls == INNER]
            hist["inner"] += np.histogram(d_in.min(), bins=edges)[0]
            inner_d.extend(d_in.tolist())
            # angle between each inner-sphere atom and the open site that holds it
            n_site = int(s.T.nsite[snap.species[0]])
            open_dirs = snap.U[0, :n_site][snap.state[0, :n_site] == 1]
            for v in c.vec[c.cls == INNER] if len(open_dirs) else ():   # cations only
                cosang = float(np.max(open_dirs @ v) / np.linalg.norm(v))
                off_axis.append(math.degrees(math.acos(min(1.0, cosang))))
            if n_in >= 2:
                v = c.vec[c.cls == INNER][:2]
                bite.append(math.degrees(math.acos(float(v[0] @ v[1] / np.linalg.norm(v[0]) / np.linalg.norm(v[1])))))
    return {
        "species": species, "ligand": ligand_name, "ionic_strength": I,
        "edges": edges.tolist(), "hist_inner": hist["inner"].tolist(), "hist_outer": hist["outer"].tolist(),
        "populations": pop,
        "mean_outer_distance": (float(np.sum(hist["outer"] * 0.5 * (edges[1:] + edges[:-1])) / hist["outer"].sum())
                                if hist["outer"].sum() else None),
        "mean_inner_distance": float(np.mean(inner_d)) if inner_d else None,
        "sd_inner_distance": float(np.std(inner_d)) if inner_d else None,
        "mean_bite_angle_deg": float(np.mean(bite)) if bite else None,
        "rms_off_axis_deg": float(np.sqrt(np.mean(np.square(off_axis)))) if off_axis else None,
        "acceptance": {k: (None if v != v else float(v)) for k, v in s.acceptance().items()},
    }


# --------------------------------------------------------------------------
# Part 2: two facing rows
# --------------------------------------------------------------------------
def build_rows(kind: str, gap: float, rng: np.random.Generator, perpendicular: bool = False,
               pka: float = 4.3):
    """Two parallel rows of groups along z on the facing sides of two excluded
    cylinders. ``gap`` is the distance between the two layers of partner atoms
    (carboxylate oxygens or the NH2 nitrogens of guanidinium).

    Returns (ligands, box, cylinders, number of charged groups per row).
    """
    lz = ROW_GROUPS * ROW_SPACING
    box = np.array([BOX_XY, BOX_XY, lz])
    if kind == "carboxylate":
        stickout = geo.CARBOXYLATE_CO_NM * math.cos(math.radians(geo.CARBOXYLATE_OCO_DEG / 2))
    else:
        stickout = geo.GUANIDINIUM_CN_NM * 0.5
    half = 0.5 * gap + stickout          # anchor atoms sit this far from the mid-plane
    xc, yc = 0.5 * BOX_XY, 0.5 * BOX_XY
    parts, charged = [], []
    for mol, sign in ((0, -1.0), (1, +1.0)):
        anchor_x = xc + sign * half
        direction = np.array([-sign, 0.0, 0.0])           # each row faces the other
        n_charged = 0
        for g in range(ROW_GROUPS):
            pos = np.array([anchor_x, yc, (g + 0.5) * ROW_SPACING])
            if kind == "carboxylate":
                prot = bool(geo.sample_protonation(1, ROW_PH, pka, rng)[0])
                n_charged += 0 if prot else 1
                parts.append(geo.carboxylate(pos, direction, perp=[0.0, 0.0, 1.0], mol=mol, protonated=prot))
            else:
                normal = [0.0, 0.0, 1.0] if (perpendicular and mol == 1) else [0.0, 1.0, 0.0]
                n_charged += 1
                parts.append(geo.guanidinium(pos, direction, normal=normal, mol=mol))
        charged.append(n_charged)
    lig = geo.LigandSet.concat(parts)
    cyl = [[xc - half - CORE_RADIUS, yc, CORE_RADIUS], [xc + half + CORE_RADIUS, yc, CORE_RADIUS]]
    return lig, box, np.array(cyl), charged


def row_run(args):
    """One grand-canonical run between two rows. Returns bridge statistics."""
    species, kind, gap, conc, pattern, perpendicular, seed, n_sweeps, stride = args
    params = load_params()
    seq = np.random.SeedSequence([seed, pattern, int(round(gap * 1000)), int(round(conc * 1e6)),
                                  params.species_index(species), int(perpendicular)])
    rng_pattern = np.random.default_rng(np.random.SeedSequence([seed, pattern]))
    lig, box, cyl, charged = build_rows(kind, gap, rng_pattern, perpendicular,
                                        float(params.value("constants.carboxyl_pKa")))
    s = IonSampler(params, lig, box, [True] * 3, ROW_IONIC_STRENGTH, np.random.default_rng(seq),
                   bath={species: conc}, cylinders=cyl)
    an = BridgeAnalyzer(params, lig, box, [True] * 3)
    tracker = BridgeTracker(interval=stride)
    s.sweep(n_sweeps // 4)
    n_samples = n_sweeps // stride
    counts = {"inner-inner": [], "inner-outer": [], "outer-outer": []}
    by_ligands = {2: 0, 3: 0, 4: 0}      # bridges whose ion holds 2, 3, >= 4 protein groups
    spans, z_all = [], []
    n_ions = []
    for _ in range(n_samples):
        s.sweep(stride)
        bridges = an.bridges(s.snapshot())
        tracker.update(bridges)
        tally = {k: 0 for k in counts}
        for b in bridges:
            tally[b.kind] += 1
            by_ligands[min(b.n_ligands, 4)] += 1
            spans.append(b.span)
            z_all.append(b.z)
        for k in counts:
            counts[k].append(tally[k])
        n_ions.append(s.n_ions)
    tracker.finish()
    life = tracker.lifetimes(include_censored=True)
    total = np.sum([counts[k] for k in counts], axis=0)
    blocks = total[: len(total) // 10 * 10].reshape(10, -1).mean(axis=1)
    return {
        "species": species, "kind": kind, "gap": gap, "conc": conc, "pattern": pattern,
        "perpendicular": bool(perpendicular), "charged_per_row": charged,
        "bridges_per_pair": float(total.mean() / ROW_GROUPS),
        "bridges_per_pair_block_se": float(blocks.std(ddof=1) / math.sqrt(10) / ROW_GROUPS),
        "by_kind_per_pair": {k: float(np.mean(v) / ROW_GROUPS) for k, v in counts.items()},
        "by_ligands_per_pair": {str(k): v / n_samples / ROW_GROUPS for k, v in by_ligands.items()},
        "mean_span": float(np.mean(spans)) if spans else None,
        "mean_lifetime_sweeps": float(life.mean()) if len(life) else None,
        "n_bridge_events": int(len(life)),
        "mean_ions_in_box": float(np.mean(n_ions)),
        "z_hist": np.histogram(z_all, bins=ROW_GROUPS, range=(0.0, box[2]))[0].tolist(),
        "acceptance": {k: (None if v != v else float(v)) for k, v in s.acceptance().items()},
    }


def aggregate_rows(results: list[dict]) -> list[dict]:
    """Average over protonation patterns; the error is the larger of the
    pattern-to-pattern standard error and the within-run block error."""
    keyed: dict = {}
    for r in results:
        keyed.setdefault((r["species"], r["kind"], r["perpendicular"], r["conc"], r["gap"]), []).append(r)
    out = []
    for (species, kind, perp, conc, gap), rs in sorted(keyed.items()):
        vals = np.array([r["bridges_per_pair"] for r in rs])
        within = math.sqrt(np.mean([r["bridges_per_pair_block_se"] ** 2 for r in rs]) / len(rs))
        between = vals.std(ddof=1) / math.sqrt(len(rs)) if len(rs) > 1 else 0.0
        spans = [r["mean_span"] for r in rs if r["mean_span"] is not None]
        lifes = [r["mean_lifetime_sweeps"] for r in rs if r["mean_lifetime_sweeps"] is not None]
        out.append({
            "species": species, "kind": kind, "perpendicular": perp, "conc": conc, "gap": gap,
            "bridges_per_pair": float(vals.mean()), "se": float(max(within, between)),
            "by_kind_per_pair": {k: float(np.mean([r["by_kind_per_pair"][k] for r in rs]))
                                 for k in ("inner-inner", "inner-outer", "outer-outer")},
            "by_ligands_per_pair": {k: float(np.mean([r["by_ligands_per_pair"][k] for r in rs]))
                                    for k in ("2", "3", "4")},
            "mean_span": float(np.mean(spans)) if spans else None,
            "mean_lifetime_sweeps": float(np.mean(lifes)) if lifes else None,
            "n_patterns": len(rs),
        })
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


PAIR_LABEL = {
    "Ca_acetate": "Ca$^{2+}$ + acetate", "Mg_acetate": "Mg$^{2+}$ + acetate",
    "Ca_sulfate": "Ca$^{2+}$ + SO$_4^{2-}$", "Mg_sulfate": "Mg$^{2+}$ + SO$_4^{2-}$",
    "SO4_guanidinium": "SO$_4^{2-}$ + guanidinium", "SO4_ammonium": "SO$_4^{2-}$ + ammonium",
    "SO4_imidazolium": "SO$_4^{2-}$ + imidazolium",
}


def fig_binding(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=SURFACE)
    names = [r["pair"] for r in rows][::-1]
    y = np.arange(len(names))
    for yi, name in zip(y, names):
        r = next(x for x in rows if x["pair"] == name)
        col = C_ION[r["species"]]
        tgt, mod = math.log10(r["K_total_target"]), math.log10(r["K_total_model"])
        ax.plot([tgt], [yi], marker="|", ms=16, mew=2.2, color=INK, ls="none", zorder=3)
        ax.plot([mod], [yi], marker="o", ms=8, color=col, mec=SURFACE, mew=1.5, ls="none", zorder=4)
        ax.annotate(f"{mod:.2f}", (mod, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [PAIR_LABEL[n] for n in names])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("log$_{10}$ K$_{total}$ (L/mol) at the target's ionic strength")
    ax.set_title("Association constants: model (dot) against target (bar)", fontsize=11, loc="left")
    ax.set_ylim(-0.6, len(names) - 0.3)
    _style(ax, "x")
    handles = [plt.Line2D([], [], marker="|", ms=12, mew=2.2, color=INK, ls="none", label="target"),
               plt.Line2D([], [], marker="o", ms=7, color=INK2, mec=SURFACE, ls="none", label="model")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_inner_outer(rows, mc, path):
    import matplotlib.pyplot as plt

    pairs = ["Ca_acetate", "Mg_acetate", "Ca_sulfate", "Mg_sulfate"]
    fig, ax = plt.subplots(figsize=(7.6, 3.2), facecolor=SURFACE)
    y = np.arange(len(pairs))[::-1]
    for yi, name in zip(y, pairs):
        r = next(x for x in rows if x["pair"] == name)
        col = C_ION[r["species"]]
        tot = r["K_total_model"]
        parts = [("inner, one atom", r["K_mono"] / tot, 1.0, None),
                 ("inner, two atoms", r["K_multi"] / tot, 0.62, "////"),
                 ("outer", r["K_outer"] / tot, 0.22, None)]
        left = 0.0
        for label, frac, alpha, hatch in parts:
            # 2px surface gap between segments
            ax.barh(yi, frac, left=left, height=0.5, color=col, alpha=alpha, hatch=hatch,
                    edgecolor=SURFACE, linewidth=1.6)
            if frac > 0.07:
                ax.text(left + frac / 2, yi, f"{100 * frac:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=INK)
            left += frac
        tgt = r["K_inner_target"]
        if tgt is not None:
            ax.plot([tgt / r["K_total_target"]] * 2, [yi - 0.36, yi + 0.36], color=INK, lw=2.0)
    for m in mc:
        name = f"{m['species']}_acetate"
        if name in pairs and m["ligand"] == "acetate":
            pop = m["populations"]
            bound = pop["outer"] + pop["inner_mono"] + pop["inner_multi"]
            if bound:
                frac = (pop["inner_mono"] + pop["inner_multi"]) / bound
                ax.plot([frac], [y[pairs.index(name)] - 0.36], marker="^", ms=7, color=INK2,
                        mec=SURFACE, ls="none", zorder=5)
    ax.set_yticks(y, [PAIR_LABEL[n] for n in pairs])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of ion pairs")
    ax.set_title("Inner- and outer-sphere shares of one ion pair", fontsize=11, loc="left")
    _style(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=1.0, label="inner sphere, monodentate"),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.62, hatch="////", edgecolor=SURFACE,
                             label="inner sphere, bidentate"),
               plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.22, label="outer sphere"),
               plt.Line2D([], [], color=INK, lw=2.0, label="target inner share"),
               plt.Line2D([], [], marker="^", ms=7, color=INK2, ls="none",
                          label="Monte Carlo, I = 0.1 M")]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_distances(mc, params, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mc), figsize=(3.6 * len(mc), 3.1), facecolor=SURFACE, sharey=False)
    for ax, m in zip(np.atleast_1d(axes), mc):
        edges = np.array(m["edges"])
        mid = 0.5 * (edges[1:] + edges[:-1])
        col = C_ION[m["species"]]
        hi, ho = np.array(m["hist_inner"]), np.array(m["hist_outer"])
        norm = max((hi + ho).sum() * (edges[1] - edges[0]), 1e-12)
        ax.fill_between(mid, hi / norm, step="mid", color=col, alpha=0.95, lw=0, label="inner sphere")
        ax.fill_between(mid, ho / norm, step="mid", color=col, alpha=0.28, lw=0, label="outer sphere")
        partner = "O" if m["ligand"] == "acetate" else "N"
        ax.set_xlabel(f"distance from ion centre to nearest {partner} (nm)")
        ax.set_title(f"{ION_LABEL[m['species']]} + {m['ligand']}", fontsize=10, loc="left")
        ax.set_xlim(0.15, 0.75)
        ax.set_ylim(bottom=0)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("probability density (1/nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ramp(color, n):
    """Single-hue sequential ramp, light to dark, for a magnitude (concentration)."""
    import matplotlib.colors as mcolors

    base = np.array(mcolors.to_rgb(color))
    white = np.array(mcolors.to_rgb(SURFACE))
    return [tuple(white + (base - white) * f) for f in np.linspace(0.38, 1.0, n)]


def fig_bridges(agg, path):
    import matplotlib.pyplot as plt

    panels = [("Ca", "carboxylate", False, "Ca$^{2+}$ between carboxylate rows"),
              ("Mg", "carboxylate", False, "Mg$^{2+}$ between carboxylate rows"),
              ("SO4", "guanidinium", True, "SO$_4^{2-}$ between guanidinium rows")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), facecolor=SURFACE, sharey=False)
    for ax, (species, kind, perp, title) in zip(axes, panels):
        rows = [a for a in agg if a["species"] == species and a["kind"] == kind and a["perpendicular"] == perp]
        concs = sorted({a["conc"] for a in rows})
        for conc, col in zip(concs, _ramp(C_ION[species], len(concs))):
            sel = sorted((a for a in rows if a["conc"] == conc), key=lambda a: a["gap"])
            x = [a["gap"] for a in sel]
            yv = np.array([a["bridges_per_pair"] for a in sel])
            se = np.array([a["se"] for a in sel])
            ax.fill_between(x, yv - se, yv + se, color=col, alpha=0.25, lw=0)
            ax.plot(x, yv, color=col, lw=2.0, marker="o", ms=4.5, mec=SURFACE, mew=0.8,
                    label=f"{1000 * conc:g} mM")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("gap between the two rows (nm)")
        ax.set_ylim(bottom=0)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, title="bath", title_fontsize=8)
    axes[0].set_ylabel("bridges per facing pair of groups")
    fig.suptitle("Bridge probability against gap and bath concentration (pH 4.2, I = 0.1 M)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_bridge_kinds(agg, conc, path):
    import matplotlib.pyplot as plt

    panels = [("Ca", "carboxylate", False, "Ca$^{2+}$, carboxylate rows"),
              ("Mg", "carboxylate", False, "Mg$^{2+}$, carboxylate rows"),
              ("SO4", "guanidinium", True, "SO$_4^{2-}$, planes perpendicular"),
              ("SO4", "guanidinium", False, "SO$_4^{2-}$, planes parallel")]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), facecolor=SURFACE)
    for ax, (species, kind, perp, title) in zip(axes, panels):
        sel = sorted((a for a in agg if a["species"] == species and a["kind"] == kind
                      and a["perpendicular"] == perp and abs(a["conc"] - conc) < 1e-12),
                     key=lambda a: a["gap"])
        x = [a["gap"] for a in sel]
        for k, marker in (("inner-inner", "o"), ("inner-outer", "s"), ("outer-outer", "^")):
            ax.plot(x, [a["by_kind_per_pair"][k] for a in sel], color=C_KIND[k], lw=2.0, marker=marker,
                    ms=5, mec=SURFACE, mew=0.8, label=k)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("gap between the two rows (nm)")
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].set_ylabel("bridges per facing pair of groups")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"Bridges by contact class on each side, bath {1000 * conc:g} mM "
                 "(for sulfate, 'inner' = direct hydrogen bond)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_markdown(rows, cf, mc, check_rows, path):
    lines = ["# S3 binding table", "",
             "Association constants in L/mol; `I` is the ionic strength (mol/L) at which the target is",
             "defined and the model evaluated. Free energy is -ln(K_total * 1 mol/L) in kT.", "",
             "| pair | I | K_total target | K_total model | K_inner target | K_inner model | monodentate | multidentate | outer | inner share | dG (kT) | K_total at I = 0.1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kin = "-" if r["K_inner_target"] is None else f"{r['K_inner_target']:.2f}"
        lines.append(
            f"| {r['pair']} | {r['ionic_strength']:.2f} | {r['K_total_target']:.2f} | {r['K_total_model']:.2f} "
            f"| {kin} | {r['K_inner_model']:.2f} | {r['K_mono']:.2f} | {r['K_multi']:.2f} | {r['K_outer']:.2f} "
            f"| {r['inner_fraction']:.2f} | {r['dG_total_kT']:.2f} | {r['K_total_at_I_0.1']:.2f} |")
    lines += ["", "For the sulfate-donor pairs 'inner' means a direct hydrogen bond and is reported in the",
              "K_inner model column; there is no separate target for it.", "",
              "Counterfactual: Mg2+ with the contact energy of Ca2+ and its own desolvation cost has an",
              f"inner-sphere share with acetate of {cf['inner_fraction']:.1e} (K_inner = {cf['K_inner']:.1e} L/mol).",
              "", "## Measured constants the model was not fitted to", "",
              "| check | I | measured log K (range) | model log K |", "|---|---|---|---|"]
    for c in check_rows:
        model = "; ".join(f"{k[12:] or 'model'} {v:.2f}".replace("_", " ").strip()
                          for k, v in c.items() if k.startswith("log_K_total"))
        lines.append(f"| {c['check']} | {c['ionic_strength']:.1f} | {c['measured']:.2f} "
                     f"({c['lo']:.2f}-{c['hi']:.2f}) | {model} |")
    lines += ["", "## Contact geometry from Monte Carlo (one ion, one ligand, I = 0.1 mol/L)", "",
              "Distances are from the ion centre (S for sulfate) to the partner atom (O or N).",
              "'Inner' is inner-sphere for the cations and a direct hydrogen bond for sulfate; the bite",
              "angle is atom-ion-atom when two partner atoms are held.", "",
              "| ion | ligand | mean inner distance (nm) | s.d. (nm) | mean outer distance (nm) | rms off-axis angle (deg) | mean bite angle (deg) | unbound | outer | inner, one atom | inner, two atoms |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in mc:
        p = m["populations"]
        n = sum(p.values())
        f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
        lines.append(f"| {m['species']} | {m['ligand']} | {f(m['mean_inner_distance'], '.3f')} "
                     f"| {f(m['sd_inner_distance'], '.3f')} | {f(m['mean_outer_distance'], '.3f')} "
                     f"| {f(m['rms_off_axis_deg'], '.1f')} "
                     f"| {f(m['mean_bite_angle_deg'], '.1f')} "
                     f"| {p['unbound'] / n:.3f} | {p['outer'] / n:.3f} | {p['inner_mono'] / n:.4f} "
                     f"| {p['inner_multi'] / n:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S3 validation of the explicit-ion model")
    ap.add_argument("--quick", action="store_true", help="short smoke run")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    params = load_params()
    quick = args.quick

    n_int = 200_000 if quick else 2_000_000
    rows = binding_table(params, args.seed, n_int)
    cf = counterfactual(params, args.seed, n_int)
    check_rows = cal.checks(params, args.seed, n_int)

    pair_jobs = [(sp, lg, args.seed + i, 4000 if quick else 60000, 20)
                 for i, (sp, lg) in enumerate([("Ca", "acetate"), ("Mg", "acetate"), ("SO4", "methylguanidinium")])]
    gaps = [0.45, 0.65, 0.85, 1.05] if quick else [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                                                      0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30]
    concs = [0.005, 0.05] if quick else [0.001, 0.005, 0.020, 0.050]
    patterns = [0] if quick else [0, 1, 2, 3]
    n_sweeps, stride = (4000, 10) if quick else (40000, 10)
    row_jobs = []
    for species in ("Ca", "Mg"):
        for gap in gaps:
            for conc in concs:
                for pat in patterns:
                    row_jobs.append((species, "carboxylate", gap, conc, pat, False, args.seed, n_sweeps, stride))
    for perp in (True, False):
        for gap in gaps:
            for conc in concs:
                row_jobs.append(("SO4", "guanidinium", gap, conc, 0, perp, args.seed, n_sweeps, stride))

    single_pair_run((pair_jobs[0][0], pair_jobs[0][1], 0, 2, 1))   # compile before forking
    with mp.get_context("fork").Pool(args.workers) as pool:
        mc = pool.map(single_pair_run, pair_jobs)
        row_results = pool.map(row_run, row_jobs, chunksize=1)
    agg = aggregate_rows(row_results)

    (args.out / "s3_binding_table.json").write_text(json.dumps(
        {"seed": args.seed, "quick": quick, "pairs": rows, "not_fitted_checks": check_rows,
         "mg_with_ca_contact_energy": cf, "single_pair_monte_carlo": mc}, indent=1), encoding="utf-8")
    write_markdown(rows, cf, mc, check_rows, args.out / "s3_binding_table.md")
    (args.out / "s3_bridges.json").write_text(json.dumps(
        {"seed": args.seed, "quick": quick, "row_spacing_nm": ROW_SPACING, "groups_per_row": ROW_GROUPS,
         "ionic_strength": ROW_IONIC_STRENGTH, "pH": ROW_PH,
         "pKa": float(params.value("constants.carboxyl_pKa")),
         "aggregated": agg, "runs": row_results}, indent=1), encoding="utf-8")
    fig_binding(rows, args.out / "s3_binding_constants.png")
    fig_inner_outer(rows, mc, args.out / "s3_inner_outer.png")
    fig_distances(mc, params, args.out / "s3_distances.png")
    fig_bridges(agg, args.out / "s3_bridges_vs_gap.png")
    fig_bridge_kinds(agg, max(concs), args.out / "s3_bridge_kinds.png")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
