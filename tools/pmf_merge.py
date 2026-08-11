"""
Delta-learning merge: fold the measured segment-pair mean forces into the
registry tables as an additive correction channel.

Pipeline:
  1. Baseline-subtract (far-gap rows remove cut-terminus artifacts).
  2. Forward-model each pose from the analytic tables at reference epsilons
     (rig mapping: phiA = -alpha, phiB = pi - beta) and fit the single
     contact-pair scale n_pairs on the well families.
  3. Fit each stagger family's force residuals to the derivative of a
     Gaussian well (position, depth, width) - robust against per-pose noise -
     then spread depths over facing angles with a wide angular kernel.
  4. Emit assets/correction2d.bin: C(delta, phiA, phiB) on the PRF2 grid,
     zero far from measured families (the analytic prior carries there).
  5. Fit the radial envelope (attR0, attW) from the gap scan.

Correction is calibrated at reference eps (el 2.5, hy 1.0) and applied in
the engines as G += epsCorr * C with epsCorr default 1.
"""
import collections
import csv
import json
import math
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
PD = os.path.join(ROOT, "pmf_data")

EPS_EL_REF = 2.5
EPS_HY_REF = 1.0
ATT_R0 = 1.5
ATT_W = 0.9
SEG_NM = 4.0
D_NM = 66.88


def load_prf2():
    with open(os.path.join(ASSETS, "profiles2d.bin"), "rb") as fh:
        assert fh.read(4) == b"PRF2"
        ver, nD, nPhi, nR = struct.unpack("<4I", fh.read(16))
        dstep, L, D, S0 = struct.unpack("<4f", fh.read(16))
        par = np.fromfile(fh, dtype="<f4", count=nD * nPhi * nPhi * 2)
    par = par.reshape(nD, nPhi, nPhi, 2)
    return par, nD, nPhi, dstep, L, D


def phi_index(phi, nPhi):
    """continuous bin coordinate for angle phi in [-pi, pi) periodic"""
    return (phi + math.pi) / (2 * math.pi) * nPhi - 0.5


def sample_G(par, dstep, delta, pa, pb):
    """G = eps_el*el + eps_hy*hy with linear interpolation on the table."""
    nD, nPhi = par.shape[0], par.shape[1]
    z = delta / dstep + (nD - 1) * 0.5
    iz = int(np.clip(math.floor(z), 0, nD - 2))
    fz = z - iz
    out = 0.0
    for (ia, wa) in _lin(phi_index(pa, nPhi), nPhi):
        for (ib, wb) in _lin(phi_index(pb, nPhi), nPhi):
            v0 = par[iz, ia, ib]
            v1 = par[iz + 1, ia, ib]
            el = v0[0] * (1 - fz) + v1[0] * fz
            hy = v0[1] * (1 - fz) + v1[1] * fz
            out += wa * wb * (EPS_EL_REF * el + EPS_HY_REF * hy)
    return out


def _lin(x, n):
    i0 = int(math.floor(x)) % n
    f = x - math.floor(x)
    return [(i0, 1 - f), ((i0 + 1) % n, f)]


def dGdD(par, dstep, delta, pa, pb):
    return (sample_G(par, dstep, delta + dstep, pa, pb) -
            sample_G(par, dstep, delta - dstep, pa, pb)) / (2 * dstep)


def env(d):
    return math.exp(-((d - ATT_R0) / ATT_W) ** 2)


def denv(d):
    return env(d) * (-2 * (d - ATT_R0) / ATT_W ** 2)


def main():
    par, nD, nPhi, dstep, L, D = load_prf2()

    rows = list(csv.DictReader(open(os.path.join(PD, "forces.csv"))))
    base = {}
    for r in rows:
        if float(r["gap"]) > 5.0:
            base[(float(r["sA"]), round(float(r["delta"]), 2))] = (
                float(r["Fx"]), float(r["Fz"]))
    meas = []
    for r in rows:
        gap = float(r["gap"])
        if gap > 5.0:
            continue
        key = (float(r["sA"]), round(float(r["delta"]), 2))
        b = base.get(key, (0.0, 0.0))
        meas.append({
            "sA": float(r["sA"]), "delta": float(r["delta"]), "gap": gap,
            "alpha": float(r["alpha"]), "beta": float(r["beta"]),
            "Fx": float(r["Fx"]) - b[0], "Fz": float(r["Fz"]) - b[1],
            "pa": -float(r["alpha"]),
            "pb": math.pi - float(r["beta"]),
        })
    print(f"{len(meas)} baseline-corrected contact measurements")

    # ---- step 2: family-mean force profiles (the statistically solid object)
    # and their 3-parameter fits: F(delta) = tilt + amp * d/dDelta(gaussian).
    # Per-pose model correlation is ~zero (the reason we measured), so the
    # merge corrects at family level: window depth -> per-engine-segment.
    SEG_ENGINE = 3.06                   # engine segment length, nm
    WINDOW_SEGS = SEG_NM / SEG_ENGINE   # measured window in engine segments
    fams = {}
    for k in (1, 2, 3):
        pts = [m for m in meas
               if abs(m["gap"] - 1.5) < 0.01 and
               abs(m["delta"] - k * D_NM) < 4.6]
        if len(pts) < 8:
            continue
        prof = collections.defaultdict(list)
        for m in pts:
            prof[round(m["delta"], 1)].append(m["Fz"])
        ds = sorted(prof)
        fbar = [float(np.mean(prof[d])) for d in ds]
        sems = [float(np.std(prof[d]) / math.sqrt(len(prof[d]))) for d in ds]

        # fit measured family profile: tilt a + amp * (2u/w) exp(-u^2)
        best = None
        for pos in np.arange(k * D_NM - 3.0, k * D_NM + 3.01, 0.25):
            for wid in (1.5, 2.0, 2.5, 3.0):
                A = np.array([[1.0,
                               (2 * (d - pos) / wid ** 2) *
                               math.exp(-((d - pos) / wid) ** 2)]
                              for d in ds])
                coef, res, *_ = np.linalg.lstsq(A, np.array(fbar), rcond=None)
                sse = float(np.sum((A @ coef - fbar) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, pos, wid, coef[1])
        sse, pos, wid, amp = best
        # amp is the force amplitude of the well derivative for the 4 nm
        # window; well depth per window = amp (gaussian normalization), then
        # per engine segment:
        depth_meas_seg = amp / WINDOW_SEGS

        # analytic model's well depth per segment at this family, averaged
        # over the measured angle combos, same fit structure
        combos = sorted(set((round(m["pa"], 2), round(m["pb"], 2))
                            for m in pts))
        gmod = []
        for d in ds:
            gv = np.mean([env(1.5) * sample_G(par, dstep, d, ca, cb)
                          for ca, cb in combos])
            gmod.append(gv)
        # model well depth: value at pos minus family-edge mean
        gpos = np.mean([env(1.5) * sample_G(par, dstep, pos, ca, cb)
                        for ca, cb in combos])
        gedge = np.mean([env(1.5) * sample_G(par, dstep, pos + 4 * wid, ca, cb)
                         for ca, cb in combos] +
                        [env(1.5) * sample_G(par, dstep, pos - 4 * wid, ca, cb)
                         for ca, cb in combos])
        depth_model_seg = gpos - gedge      # negative = well (kT per segment)

        corr_depth = depth_meas_seg - depth_model_seg

        # bootstrap the fitted depth over pose resamples -> shrinkage
        rng = np.random.default_rng(7)
        boots = []
        for _ in range(200):
            prof_b = collections.defaultdict(list)
            for m in rng.choice(pts, size=len(pts), replace=True):
                prof_b[round(m["delta"], 1)].append(m["Fz"])
            if len(prof_b) < 4:
                continue
            dsb = sorted(prof_b)
            fbarb = [float(np.mean(prof_b[d])) for d in dsb]
            A = np.array([[1.0,
                           (2 * (d - pos) / wid ** 2) *
                           math.exp(-((d - pos) / wid) ** 2)] for d in dsb])
            coefb, *_ = np.linalg.lstsq(A, np.array(fbarb), rcond=None)
            boots.append(coefb[1] / WINDOW_SEGS)
        sem = float(np.std(boots)) if boots else abs(corr_depth)
        shrink = corr_depth ** 2 / (corr_depth ** 2 + sem ** 2)
        applied = corr_depth * shrink
        CAP = 6.0
        if abs(applied) > CAP:
            print(f"  [cap] family {k}D correction {applied:+.2f} clamped "
                  f"to +/-{CAP} kT/seg (needs replicates to trust deeper)")
            applied = math.copysign(CAP, applied)
        fams[k] = {"pos": pos, "wid": wid, "meas": depth_meas_seg,
                   "model": depth_model_seg, "raw": corr_depth, "sem": sem,
                   "corr": applied, "n": len(pts)}
        print(f"family {k}D: well {pos:.1f} nm (w {wid:.1f}) | measured "
              f"{depth_meas_seg:+.2f} kT/seg vs model {depth_model_seg:+.2f} "
              f"| raw corr {corr_depth:+.2f} +/- {sem:.2f} -> applied "
              f"{applied:+.2f} (shrink {shrink:.2f})")

    # ---- step 4: correction grid C(delta) uniform in facing angles (the
    # angular data is sequence-confounded at n=4/cell; v1 stays isotropic)
    corr = np.zeros((nD, nPhi, nPhi), dtype=np.float32)
    izc = (nD - 1) // 2
    for k, f in fams.items():
        for sgn in (+1, -1):
            pos = sgn * f["pos"]
            lo = max(0, int((pos - 4 * f["wid"]) / dstep + izc))
            hi = min(nD, int((pos + 4 * f["wid"]) / dstep + izc) + 1)
            for iz in range(lo, hi):
                dz = (iz - izc) * dstep
                g = math.exp(-((dz - pos) / f["wid"]) ** 2)
                corr[iz, :, :] += f["corr"] * g

    with open(os.path.join(ASSETS, "correction2d.bin"), "wb") as fh:
        fh.write(b"CORR")
        np.array([1, nD, nPhi, 0], dtype="<u4").tofile(fh)
        np.array([dstep, L, D, 0.0], dtype="<f4").tofile(fh)
        corr.astype("<f4").tofile(fh)
    print(f"wrote correction2d.bin (range {corr.min():+.2f}..{corr.max():+.2f} "
          f"per-pair kT at reference eps)")

    # ---- step 5: radial envelope fit from the gap scan
    gaps = collections.defaultdict(list)
    for m in meas:
        if abs(m["delta"] - D_NM) < 0.5 and m["alpha"] == 0 and m["beta"] == 0:
            gaps[m["gap"]].append(m["Fx"])
    gpts = sorted((g, sum(v) / len(v)) for g, v in gaps.items())
    bestr = None
    if len(gpts) >= 3:
        Gc = sample_G(par, dstep, D_NM, 0.0, math.pi)
        for r0 in np.arange(1.2, 1.9, 0.05):
            for w in np.arange(0.6, 1.6, 0.05):
                sse = 0.0
                for g, fx in gpts:
                    e = math.exp(-((g - r0) / w) ** 2)
                    de = e * (-2 * (g - r0) / w ** 2)
                    model = WINDOW_SEGS * de * Gc
                    sse += (fx - model) ** 2
                if bestr is None or sse < bestr[0]:
                    bestr = (sse, r0, w)
        print(f"radial fit: attR0 = {bestr[1]:.2f} nm, attW = {bestr[2]:.2f} nm "
              f"(was {ATT_R0}, {ATT_W}) from {len(gpts)} gap points")

    with open(os.path.join(PD, "merge_report.txt"), "w") as fh:
        fh.write(f"window/engine-segment ratio: {WINDOW_SEGS:.2f}\n")
        for k, f in fams.items():
            fh.write(f"family {k}D: pos {f['pos']:.1f} width {f['wid']:.1f} "
                     f"measured {f['meas']:+.2f} model {f['model']:+.2f} "
                     f"raw {f['raw']:+.2f}+/-{f['sem']:.2f} applied "
                     f"{f['corr']:+.2f} kT/seg n {f['n']}\n")
        if bestr:
            fh.write(f"radial: attR0 {bestr[1]:.2f} attW {bestr[2]:.2f}\n")

    # validation plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, k in enumerate(sorted(fams)):
        ax = axes[i]
        f = fams[k]
        pts = [m for m in meas if abs(m["gap"] - 1.5) < 0.01 and
               abs(m["delta"] - k * D_NM) < 4.6]
        combos = sorted(set((round(m["pa"], 2), round(m["pb"], 2)) for m in pts))
        ds = sorted(set(round(m["delta"], 1) for m in pts))
        meas_avg = [np.mean([m["Fz"] for m in pts if round(m["delta"], 1) == d])
                    for d in ds]

        def modelF(d, with_corr):
            h = 0.25
            def E(dd):
                e = np.mean([env(1.5) * sample_G(par, dstep, dd, ca, cb)
                             for ca, cb in combos])
                if with_corr:
                    e += f["corr"] * math.exp(-((dd - f["pos"]) / f["wid"]) ** 2)
                return e
            return -WINDOW_SEGS * (E(d + h) - E(d - h)) / (2 * h)

        ax.plot(ds, meas_avg, "ko-", label="measured (family mean)")
        ax.plot(ds, [modelF(d, False) for d in ds], "b.--", label="prior")
        ax.plot(ds, [modelF(d, True) for d in ds], "r.-", label="prior+corr")
        ax.axvline(k * D_NM, color="gray", ls=":")
        ax.set_title(f"{k}D family (corr {f['corr']:+.2f} kT/seg)")
        ax.set_xlabel("stagger (nm)")
        if i == 0:
            ax.set_ylabel("Fz (kT/nm)")
            ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "validation", "merge_fit.png"), dpi=130)
    print("wrote merge_report.txt + validation/merge_fit.png")


if __name__ == "__main__":
    main()
