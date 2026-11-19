"""Measured against predicted: tables and figures for docs/explicit_solvent_pmf.md.

``python -m aamd.compare``

Reads every finished PMF (umbrella windows in aamd/work/SYSTEM/) and every
prediction (aamd/results/predict_SYSTEM.json) and writes:

  aamd/results/pmf_SYSTEM.json      the analysed PMF (MBAR, WHAM check, block
                                    errors, halves)
  aamd/results/comparison.csv       one row per system
  aamd/results/comparison.md        the same as a Markdown table
  aamd/results/fig_pmf_SYSTEM.png   PMF with errors and halves, coarse F_free
  aamd/results/fig_summary.png      contact free energies, measured vs coarse

Definitions (all kT at 298 K, relative to the mean over 2.55-2.80 nm):

  dG_contact   minimum of the measured PMF, with its block error
  d_contact    where it lies
  F_coarse     the coarse free-rotation energy at d_contact (``predict.F_free``)
               and its electrostatic and hydrophobic parts
  residual     dG_contact - F_coarse: what a correction channel would have to add
  eps_h_fit    the hydrophobic strength that would make the coarse model match:
               F_el + eps_h * H = dG_contact, H the rotation-averaged hydrophobic
               overlap at d_contact; least squares over the monovalent systems
  eps_h_offset the same with a register-independent constant c added,
               dG_contact - E_el = -eps_h H + c, weighted by the block errors:
               separates a stagger-specific (hydrophobic) attraction from a
               nonspecific one
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import build, pmf
from .umbrella import PROTOCOL

RESULTS = build.AAMD / "results"
EDGES = np.arange(1.20, 2.951, 0.02)
REF = (2.55, 2.80)


def analysed_pmf(name):
    d = build.WORK / name
    win = pmf.load_windows(d, PROTOCOL.centres, PROTOCOL.k_umb)
    runs = pmf.window_dirs(d, PROTOCOL.centres)
    if len(win.series) < len(runs) or not all((r / "done.json").exists() for _, r in runs):
        return None
    res = pmf.analyse(win, EDGES, ref=REF)
    pmf.save(res, RESULTS / f"pmf_{name}.json",
             meta={"ns_per_window": float(np.mean([len(s) for s in win.series])) / 1000.0,
                   "n_windows": len(win.series), "k_umb_kj_nm2": PROTOCOL.k_umb})
    return res


R_NOMINAL_NM = 0.75      # modcoll's nominal molecular radius; surface separation = d - 2 R


def short_range(res, sysdir, centres):
    """The part of a PMF at sub-nanometre surface separation, and the waters.

    * wall: the separation inside the minimum where F has risen 2 kT above it;
    * decay length: exponential fit F - F_min = A exp(-(d - d_min)/lambda) to
      the inner branch (points from the minimum inward with F - F_min > 0.2 kT),
      when there are at least three such points;
    * long-range attraction: the mean of F over surface separations 0.5-1.0 nm
      (d = 2.0-2.5 nm), i.e. what is left of the attraction beyond the first
      hydration layers;
    * bridging waters per window (water O within 0.35 nm of both segments),
      where recorded.
    """
    k = int(np.nanargmin(res.F))
    d_min, f_min = float(res.x[k]), float(res.F[k])
    inner = [(float(x), float(f - f_min)) for x, f in zip(res.x[:k + 1][::-1], res.F[:k + 1][::-1])
             if np.isfinite(f)]
    wall = next((x for x, df in inner if df >= 2.0), None)
    pts = [(x, df) for x, df in inner if df > 0.2]
    decay = None
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts])
        y = np.log(np.array([p[1] for p in pts]))
        slope, icpt = np.polyfit(x - d_min, y, 1)
        if slope < 0:
            decay = {"lambda_nm": float(-1.0 / slope), "A_kT": float(np.exp(icpt)), "n_points": len(pts)}
    m = (res.x >= 2.0) & (res.x <= 2.5) & np.isfinite(res.F)
    water = []
    for c, wdir in pmf.window_dirs(sysdir, centres):
        files = sorted(wdir.glob("hyd_*.npy"))
        cvf = sorted(wdir.glob("cv_*.npy"))
        if files and cvf:
            h = np.concatenate([np.load(f) for f in files])
            cv = np.concatenate([np.load(f) for f in cvf])
            water.append({"r0": float(c), "d_mean": float(cv[:, 1].mean()), "bridging": float(h[:, 1].mean()),
                          "bridging_sd": float(h[:, 1].std()), "n": int(len(h))})
    return {"d_min_nm": d_min, "surface_sep_min_nm": d_min - 2 * R_NOMINAL_NM, "F_min_kT": f_min,
            "wall_2kT_nm": wall, "inner_decay": decay,
            "F_mean_surface_0.5_1.0nm_kT": float(np.mean(res.F[m])) if m.any() else None,
            "bridging_waters": water}


def interp(pred, key, d):
    return float(np.interp(d, pred["d_nm"], pred[key]))


def row_for(name, res, pred):
    xmin, fmin, dfmin = res.minimum()
    k = int(np.nanargmin(res.F))
    half = float(abs(res.F_first[k] - res.F_second[k]))
    r = {"system": name, "dG_contact_kT": fmin, "err_block_kT": dfmin, "halves_diff_kT": half,
         "d_contact_nm": xmin, "dG_wham_kT": float(res.F_wham[k])}
    if pred:
        r.update({
            "F_coarse_kT": interp(pred, "F_free", xmin),
            "F_coarse_el_kT": interp(pred, "F_free_el", xmin),
            "F_coarse_hy_kT": interp(pred, "F_free_hy", xmin),
            "F_coarse_at_1.65_kT": interp(pred, "F_free", 1.65),
            "F_coarse_min_kT": float(np.min(pred["F_free"])),
            "H_overlap": interp(pred, "hydrophobic_overlap", xmin),
            "E_mean_el_kT": interp(pred, "E_mean_el", xmin),
            "net_charge_A_B": pred["net_charge"],
        })
        r["residual_kT"] = r["dG_contact_kT"] - r["F_coarse_kT"]
    return r


def fit_eps_h(rows):
    """Least-squares eps_h from dG = E_el + eps_h * (-H) over the given rows."""
    y = np.array([r["dG_contact_kT"] - r["E_mean_el_kT"] for r in rows])
    x = np.array([-r["H_overlap"] for r in rows])
    if len(rows) < 2 or not np.any(x):
        return None
    eps = float(x @ y / (x @ x))
    resid = y - eps * x
    se = float(np.sqrt(resid @ resid / max(len(rows) - 1, 1) / (x @ x)))
    return {"eps_h_fit": eps, "se": se, "n": len(rows), "eps_h_model": 0.05,
            "note": "rotation-averaged energies; eps_h in kT per KD unit^2"}


def fit_eps_h_offset(rows):
    """Weighted least squares for dG - E_el = -eps_h H + c (block errors as sigma)."""
    if len(rows) < 3:
        return None
    y = np.array([r["dG_contact_kT"] - r["E_mean_el_kT"] for r in rows])
    X = np.column_stack([[-r["H_overlap"] for r in rows], np.ones(len(rows))])
    w = 1.0 / np.array([max(r["err_block_kT"], 1e-3) for r in rows]) ** 2
    A = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    dof = len(rows) - 2
    cov = np.linalg.inv(A) * max(chi2 / dof, 1.0)   # inflate if the scatter exceeds the errors
    return {"eps_h": float(beta[0]), "eps_h_se": float(np.sqrt(cov[0, 0])), "c_kT": float(beta[1]),
            "c_se": float(np.sqrt(cov[1, 1])), "chi2": chi2, "dof": dof, "n": len(rows)}


def figure(name, res, pred):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    m = np.isfinite(res.F)
    ax.fill_between(res.x[m], (res.F - 2 * res.dF_block)[m], (res.F + 2 * res.dF_block)[m],
                    color="#2a6fdb", alpha=0.18, lw=0, label="± 2 block standard errors")
    ax.plot(res.x[m], res.F[m], color="#2a6fdb", lw=2, label="explicit solvent PMF")
    ax.plot(res.x, res.F_first, color="#2a6fdb", lw=0.8, ls="--", label="first half")
    ax.plot(res.x, res.F_second, color="#2a6fdb", lw=0.8, ls=":", label="second half")
    if pred:
        ax.plot(pred["d_nm"], pred["F_free"], color="#d9480f", lw=1.8, label="coarse site model, free rotation")
        ax.plot(pred["d_nm"], pred["F_free_hy"], color="#d9480f", lw=0.8, ls="--", label="coarse, hydrophobic part")
        ax.plot(pred["d_nm"], pred["F_free_el"], color="#d9480f", lw=0.8, ls=":", label="coarse, electrostatic part")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(1.65, color="0.75", lw=0.6, ls="--")
    ax.set_xlim(1.2, 2.9)
    ax.set_xlabel("inter-axial separation (nm)")
    ax.set_ylabel("free energy (kT)")
    ax.set_title(name)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_pmf_{name}.png", dpi=130)
    plt.close(fig)


def summary_figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["dG_contact_kT"] for r in rows], 0.4, yerr=[2 * r["err_block_kT"] for r in rows],
           color="#2a6fdb", label="explicit solvent, PMF minimum")
    ax.bar(x + 0.2, [r.get("F_coarse_kT", np.nan) for r in rows], 0.4, color="#d9480f",
           label="coarse site model at the same separation")
    ax.set_xticks(x, [r["system"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("contact free energy (kT)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_summary.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    RESULTS.mkdir(exist_ok=True)
    rows = []
    hydration = {}
    for name in build.SYSTEMS:
        if not (build.WORK / name / "pull.json").exists():
            continue
        res = analysed_pmf(name)
        if res is None:
            continue
        pf = RESULTS / f"predict_{name}.json"
        pred = json.loads(pf.read_text()) if pf.exists() else None
        rows.append(row_for(name, res, pred))
        hydration[name] = short_range(res, build.WORK / name, PROTOCOL.centres)
        figure(name, res, pred)
    if not rows:
        print("no finished PMF yet")
        return
    mono = [r for r in rows if r["system"].count("-") == 1 and "H_overlap" in r]
    fit = fit_eps_h(mono)
    fit_c = fit_eps_h_offset(mono)
    by = {r["system"]: r for r in rows}
    derived = {}
    for chem in ("native", "succ", "acet"):
        a, b = by.get(f"{chem}-1D"), by.get(f"{chem}-05D")
        if a and b:
            derived[f"{chem}: G(1D) - G(0.5D)"] = {
                "measured": a["dG_contact_kT"] - b["dG_contact_kT"],
                "err": float(np.hypot(a["err_block_kT"], b["err_block_kT"])),
                "coarse": a.get("F_coarse_kT", np.nan) - b.get("F_coarse_kT", np.nan)}
    for ion in ("Ca", "Mg", "SO4"):
        for chem in ("native", "succ"):
            a, b = by.get(f"{chem}-1D-{ion}"), by.get(f"{chem}-1D")
            if a and b:
                derived[f"{chem}-1D: G({ion}) - G(monovalent)"] = {
                    "measured": a["dG_contact_kT"] - b["dG_contact_kT"],
                    "err": float(np.hypot(a["err_block_kT"], b["err_block_kT"])),
                    # mean-field screening only; ion bridges come from aamd.ions_compare
                    "coarse": a.get("F_coarse_kT", np.nan) - b.get("F_coarse_kT", np.nan)}
    if by.get("succ-1D") and by.get("native-1D"):
        a, b = by["succ-1D"], by["native-1D"]
        derived["1D: G(succ, pH 4.2) - G(native, pH 7.4)"] = {
            "measured": a["dG_contact_kT"] - b["dG_contact_kT"],
            "err": float(np.hypot(a["err_block_kT"], b["err_block_kT"])),
            "coarse": a.get("F_coarse_kT", np.nan) - b.get("F_coarse_kT", np.nan)}
    keys = list(rows[0].keys())
    with open(RESULTS / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in keys, k)))
        w.writeheader()
        w.writerows(rows)
    lines = ["| system | dG_contact (kT) | d (nm) | halves diff | coarse F at d | coarse el | coarse hy | residual |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['system']} | {r['dG_contact_kT']:+.2f} +- {r['err_block_kT']:.2f} | {r['d_contact_nm']:.2f} | "
                     f"{r['halves_diff_kT']:.2f} | {r.get('F_coarse_kT', np.nan):+.2f} | {r.get('F_coarse_el_kT', np.nan):+.2f} | "
                     f"{r.get('F_coarse_hy_kT', np.nan):+.2f} | {r.get('residual_kT', np.nan):+.2f} |")
    lines += ["", "| difference | measured (kT) | coarse (kT) |", "|---|---|---|"]
    for k, v in derived.items():
        lines.append(f"| {k} | {v['measured']:+.2f} +- {v['err']:.2f} | {v.get('coarse', float('nan')):+.2f} |")
    if fit:
        lines += ["", f"eps_h that would match the monovalent systems: {fit['eps_h_fit']:.3f} +- {fit['se']:.3f} "
                      f"kT/KD^2 (n = {fit['n']}; model value 0.05)"]
    if fit_c:
        lines += [f"with a register-independent offset c: eps_h = {fit_c['eps_h']:.3f} +- {fit_c['eps_h_se']:.3f} "
                  f"kT/KD^2, c = {fit_c['c_kT']:.2f} +- {fit_c['c_se']:.2f} kT "
                  f"(chi2 {fit_c['chi2']:.1f} on {fit_c['dof']} dof)"]
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    lines += ["", "| system | minimum at d (surface sep.) | F 2 kT above minimum, inside, at | inner decay length | "
                  "mean F at surface sep. 0.5-1.0 nm |", "|---|---|---|---|---|"]
    for n, h in hydration.items():
        dec = h["inner_decay"]
        f_lr = h["F_mean_surface_0.5_1.0nm_kT"]
        lines.append(f"| {n} | {h['d_min_nm']:.2f} nm ({h['surface_sep_min_nm']:.2f} nm) | "
                     f"{h['wall_2kT_nm'] if h['wall_2kT_nm'] is not None else 'not reached'} | "
                     f"{('%.3f nm (%d points)' % (dec['lambda_nm'], dec['n_points'])) if dec else '-'} | "
                     f"{('%+.2f kT' % f_lr) if f_lr is not None else '-'} |")
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "comparison.json").write_text(json.dumps({"rows": rows, "derived": derived, "eps_h_fit": fit,
                                                         "eps_h_offset_fit": fit_c,
                                                         "short_range": hydration}, indent=1))
    summary_figure(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
