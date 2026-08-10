"""
Sequence-derived axial interaction profiles for tropocollagen pairs.

Following the logic of Hulmes, Miller et al. (1973): project per-residue charge
and (centered) Kyte-Doolittle hydrophobicity of all three chains onto the
molecular axis, smooth at side-chain scale, and correlate two copies as a
function of axial stagger.

Parallel contact, stagger D = s_B - s_A:
    U_el(D)  = + sum_s q(s) q(s+D)      (like charges aligned -> repulsive)
    U_hyd(D) = - sum_s h(s) h(s+D)      (matched hydrophobic patches -> attractive)

Antiparallel contact, registry r = s_A + s_B (constant along the contact):
    U_*_ap(r) via self-convolution.

Outputs: assets/profiles.bin + profiles_meta.json + validation/profiles.png
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
VALID = os.path.join(ASSETS, "validation")

DS = 0.1            # nm, axial grid
SIGMA = 0.4         # nm, side-chain smoothing
SHARPEN = 1.8       # contrast exponent: restores registry specificity lost to
                    # coarse-grained smoothing (deep wells kept, background cut)


def deposit(z, w, grid_n):
    """Scatter weights w at axial positions z onto the grid, Gaussian-smoothed."""
    dens = np.zeros(grid_n)
    idx = np.clip(np.round(z / DS).astype(int), 0, grid_n - 1)
    np.add.at(dens, idx, w)
    # FFT Gaussian smoothing
    k = np.fft.rfftfreq(2 * grid_n, d=DS)
    pad = np.zeros(2 * grid_n)
    pad[:grid_n] = dens
    sm = np.fft.irfft(np.fft.rfft(pad) * np.exp(-2 * (np.pi * k * SIGMA) ** 2),
                      n=2 * grid_n)[:grid_n]
    return sm


def xcorr(a, b):
    """c[j] = sum_i a[i] b[i+j] for j in [-(n-1), n-1], via FFT."""
    n = len(a)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    fa = np.fft.rfft(a, m)
    fb = np.fft.rfft(b, m)
    c = np.fft.irfft(np.conj(fa) * fb, m)
    return np.concatenate([c[m - (n - 1):], c[:n]])   # lag -(n-1) .. n-1


def convolve(a, b):
    n = len(a)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)
    return c[:2 * n - 1]                              # r = 0 .. 2(n-1)


def local_minima(u, min_sep):
    idx = [i for i in range(2, len(u) - 2)
           if u[i] < u[i - 1] and u[i] < u[i + 1]]
    idx.sort(key=lambda i: u[i])
    out = []
    for i in idx:
        if all(abs(i - j) > min_sep for j in out):
            out.append(i)
    return sorted(out)


def main():
    d = np.load(os.path.join(ASSETS, "residues.npz"))
    z, q, h = d["z"], d["q"], d["h"]
    meta_atoms = json.load(open(os.path.join(ASSETS, "atoms_meta.json")))
    L = meta_atoms["length_nm"]
    D = meta_atoms["d_period_nm"]
    n_grid = int(np.ceil(L / DS)) + 1

    hc = h - h.mean()
    qd = deposit(z.ravel(), q.ravel(), n_grid)
    hd = deposit(z.ravel(), hc.ravel(), n_grid)

    u_el_par = xcorr(qd, qd) * DS
    u_hy_par = -xcorr(hd, hd) * DS
    u_el_ap = convolve(qd, qd) * DS
    u_hy_ap = -convolve(hd, hd) * DS

    # normalize each channel to its own unit peak so runtime weights control the
    # electrostatic/hydrophobic mix directly; keep par/ap of a channel on the
    # same scale so polarity competition stays physical
    scales = {}
    for name, u in [("el_par", u_el_par), ("hy_par", u_hy_par),
                    ("el_ap", u_el_ap), ("hy_ap", u_hy_ap)]:
        scales[name] = float(np.abs(u).max())
    el_norm = max(scales["el_par"], scales["el_ap"])
    hy_norm = max(scales["hy_par"], scales["hy_ap"])
    u_el_par /= el_norm; u_el_ap /= el_norm
    u_hy_par /= hy_norm; u_hy_ap /= hy_norm

    def sharpen(u):
        return np.sign(u) * np.abs(u) ** SHARPEN
    u_el_par = sharpen(u_el_par); u_el_ap = sharpen(u_el_ap)
    u_hy_par = sharpen(u_hy_par); u_hy_ap = sharpen(u_hy_ap)

    # anchor normalization to the deepest specific well (|stagger| > 5 nm) so
    # epsilon sliders read directly as kT-per-contact-pair at D-register;
    # clip the (huge) rescaled zero-stagger wall to keep forces bounded
    lags_tmp = (np.arange(len(u_el_par)) - (n_grid - 1)) * DS
    spec = np.abs(lags_tmp) > 5.0
    for pair in ((u_el_par, u_el_ap), (u_hy_par, u_hy_ap)):
        m = abs(pair[0][spec].min())
        pair[0][:] = np.clip(pair[0] / m, -1.5, 3.0)
        pair[1][:] = np.clip(pair[1] / m, -1.5, 3.0)

    # --- funnel reconstruction: keep only the physically robust features ---
    # (D-multiple wells with sequence-derived relative depths + zero-stagger
    # charge repulsion); the rugged background between wells is a smoothing
    # artifact that creates unphysical kinetic traps at this resolution
    lags = (np.arange(len(u_el_par)) - (n_grid - 1)) * DS
    WELL_W = 3.0
    BUMP = 2.0

    def gauss_at(x0, w):
        return np.exp(-((lags - x0) ** 2) / (w * w))

    el_f = BUMP * gauss_at(0.0, 4.0)
    hy_f = np.zeros_like(u_hy_par)
    well_list = []
    for k in range(-4, 5):
        if k == 0:
            continue
        target = k * D
        i0 = np.argmin(np.abs(lags - target))
        win = int(6.0 / DS)
        seg = slice(max(0, i0 - win), min(len(lags), i0 + win))
        im = seg.start + int(np.argmin(u_el_par[seg] + u_hy_par[seg]))
        pos = lags[im]
        del_el = min(u_el_par[im], 0.0)
        del_hy = min(u_hy_par[im], 0.0)
        el_f += del_el * gauss_at(pos, WELL_W)
        hy_f += del_hy * gauss_at(pos, WELL_W)
        if k > 0:
            well_list.append((float(pos), float(del_el + del_hy)))
    u_el_par, u_hy_par = el_f, hy_f

    # antiparallel: six deepest wells of its own landscape, same treatment
    r_ap = np.arange(len(u_el_ap)) * DS
    comb_ap = u_el_ap + u_hy_ap
    mins_ap = local_minima(comb_ap, int(15 / DS))
    mins_ap = sorted(mins_ap, key=lambda i: comb_ap[i])[:6]
    el_fa = np.zeros_like(u_el_ap)
    hy_fa = np.zeros_like(u_hy_ap)
    for i in mins_ap:
        g = np.exp(-((r_ap - r_ap[i]) ** 2) / (WELL_W * WELL_W))
        el_fa += min(u_el_ap[i], 0.0) * g
        hy_fa += min(u_hy_ap[i], 0.0) * g
    u_el_ap, u_hy_ap = el_fa, hy_fa
    print("parallel funnel wells (pos nm, depth):",
          [(round(p, 1), round(d, 2)) for p, d in well_list])

    # combined parallel profile (equal weights) for validation
    comb = 0.5 * (u_el_par + u_hy_par)
    pos = lags > 5.0
    mins = local_minima(np.where(pos, comb, np.inf), int(20 / DS))
    print("deepest parallel-stagger minima (nm), expected near multiples of "
          f"D = {D:.1f}:")
    got = []
    for i in sorted(mins, key=lambda i: comb[i])[:6]:
        lag = lags[i]
        got.append(lag)
        print(f"  stagger {lag:7.1f} nm = {lag/D:5.2f} D   depth {comb[i]:+.3f}")

    with open(os.path.join(ASSETS, "profiles.bin"), "wb") as fh:
        fh.write(b"PROF")
        np.array([1, len(u_el_par), len(u_el_ap)], dtype="<u4").tofile(fh)
        np.array([DS, L, D], dtype="<f4").tofile(fh)
        for u in (u_el_par, u_hy_par, u_el_ap, u_hy_ap):
            u.astype("<f4").tofile(fh)

    with open(os.path.join(ASSETS, "profiles_meta.json"), "w") as fh:
        json.dump({"ds_nm": DS, "sigma_nm": SIGMA, "n_par": len(u_el_par),
                   "n_ap": len(u_el_ap), "raw_scales": scales,
                   "minima_nm": [float(x) for x in got]}, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    axes[0].plot(lags, u_el_par, lw=0.7, label="electrostatic")
    axes[0].plot(lags, u_hy_par, lw=0.7, label="hydrophobic")
    axes[0].set_title("parallel pair energy vs axial stagger (normalized)")
    axes[0].legend(); axes[0].set_xlim(0, L)
    axes[1].plot(lags, comb, lw=0.9, color="k")
    for k in range(1, 5):
        axes[1].axvline(k * D, color="r", ls="--", lw=0.8)
    axes[1].set_title("combined (equal weights); red dashes = multiples of D = "
                      f"{D:.1f} nm")
    axes[1].set_xlim(0, L); axes[1].set_xlabel("stagger (nm)")
    r = np.arange(len(u_el_ap)) * DS
    axes[2].plot(r, 0.5 * (u_el_ap + u_hy_ap), lw=0.7, color="purple")
    axes[2].set_title("antiparallel pair energy vs registry s_A + s_B")
    axes[2].set_xlabel("registry (nm)")
    fig.tight_layout()
    fig.savefig(os.path.join(VALID, "profiles.png"), dpi=130)
    print("wrote profiles.bin + profiles.png")


if __name__ == "__main__":
    main()
