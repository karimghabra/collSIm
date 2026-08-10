"""
2D azimuthal registry potential U(stagger, facingA, facingB).

Instead of collapsing the molecular surface to a 1D axial profile (which
projects away the azimuthal escape paths and then needs artificial contrast
repair), place every residue at its true (axial s, azimuth phi) position on
the triple-helix surface and correlate two molecular surfaces as a function
of axial stagger AND the azimuth each molecule presents to the contact.

E_par(D, pa, pb) = sum_s rho_A(s, pa) * rho_B(s + D, pb)
with rho the (charge | centered hydrophobicity) surface density smoothed over
the side-chain reach in s and over the contact patch in phi.
Antiparallel: B flipped (s -> L - s, phi -> -phi).

Output: assets/profiles2d.bin
  header: magic 'PRF2', u32 version, u32 nD, u32 nPhi, u32 nR,
          f32 dsD, f32 L, f32 Dperiod
  then float32 arrays, each [n, nPhi, nPhi] with n = nD (par) or nR (ap):
  el_par, hy_par, el_ap, hy_ap
Validation: assets/validation/profiles2d.png
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
VALID = os.path.join(ASSETS, "validation")

DS = 0.25          # nm axial grid for surface density
SIG_S = 0.35       # nm axial smoothing (side-chain reach)
SIG_PHI = 0.42     # rad (~24 deg) contact-patch smoothing
NPHI = 96          # internal azimuthal resolution
NPHI_OUT = 24      # texture azimuthal resolution
DSTEP = 0.5        # nm stagger resolution of the output table

CHARGE = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.1, "ASP": -1.0, "GLU": -1.0}
KD = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
      "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9,
      "TYR": -1.3, "PRO": -1.6, "HYP": -1.6, "HIS": -3.2, "GLU": -3.5,
      "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}
BACKBONE = {"N", "CA", "C", "O"}


AA_CHARGE1 = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
AA_KD1 = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
          "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
          "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9,
          "R": -4.5}


def telopeptides(L):
    """Flanking (telopeptide) residues of both chains at extended axial
    positions off the helix ends: (s, q, h) with azimuth treated as uniform
    (flexible ends). N-side ~16 aa, C-side ~25 aa per chain, real sequences."""
    import re
    DATA = os.path.join(ROOT, "data")
    out = []
    RISE = 0.28   # nm/res, near-extended
    for fasta, copies in (("P02452.fasta", 2), ("P08123.fasta", 1)):
        seq = "".join(l.strip() for l in open(os.path.join(DATA, fasta))
                      if not l.startswith(">"))
        best = (0, 0)
        i = 0
        while i < len(seq):
            if seq[i] == "G":
                n = 0
                while i + 3 * n < len(seq) and seq[i + 3 * n] == "G":
                    n += 1
                if n > best[1]:
                    best = (i, n)
            i += 1
        start, ng = best
        end = start + 3 * ng
        ntel = seq[max(0, start - 16):start]
        ctel = seq[end:end + 25]
        for rep in range(copies):
            for j, aa in enumerate(ntel):
                s = -(len(ntel) - j) * RISE
                out.append((s, AA_CHARGE1.get(aa, 0.0), AA_KD1.get(aa, 0.0),
                            TITRATABLE1.get(aa, -1)))
            for j, aa in enumerate(ctel):
                s = L + (j + 1) * RISE
                out.append((s, AA_CHARGE1.get(aa, 0.0), AA_KD1.get(aa, 0.0),
                            TITRATABLE1.get(aa, -1)))
    return out


TITRATABLE = {"ASP": 0, "GLU": 1, "HIS": 2, "LYS": 3, "ARG": 4}
TITRATABLE1 = {"D": 0, "E": 1, "H": 2, "K": 3, "R": 4}
NTYPES = 5

MASS1 = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
         "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16,
         "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
         "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13}


def load_residue_surface():
    """(s, phi, q, h) per residue from the full-atom PDB."""
    path = os.path.join(ASSETS, "tropocollagen.pdb")
    res = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            rn = line[17:20].strip()
            cid = line[21]
            ri = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            key = (cid, ri)
            d = res.setdefault(key, {"rn": rn, "side": [], "ca": None})
            if name == "CA":
                d["ca"] = (x, y, z)
            if name not in BACKBONE:
                d["side"].append((x, y, z))
    out = []
    for (cid, ri), d in res.items():
        pts = d["side"] if d["side"] else ([d["ca"]] if d["ca"] else [])
        if not pts:
            continue
        m = np.mean(np.array(pts), axis=0)
        s = m[2] / 10.0
        phi = np.arctan2(m[1], m[0])
        q = CHARGE.get(d["rn"], 0.0)
        h = KD.get(d["rn"], 0.0)
        t = TITRATABLE.get(d["rn"], -1)
        out.append((s, phi, q, h, t))
    return np.array(out)


def deposit_2d(sv, phv, w, n_s, telo=None):
    """Surface density on (s, phi) grid, Gaussian smoothed in both dims.
    telo: list of (s, w) deposited uniformly over phi (flexible ends)."""
    grid = np.zeros((n_s, NPHI))
    si = np.clip(np.round(sv / DS).astype(int), 0, n_s - 1)
    pi = np.mod(np.round((phv + np.pi) / (2 * np.pi) * NPHI).astype(int), NPHI)
    np.add.at(grid, (si, pi), w)
    if telo:
        for s, tw in telo:
            i = int(np.clip(round(s / DS), 0, n_s - 1))
            grid[i, :] += tw / NPHI
    # smooth axially (fft, zero-padded) and azimuthally (fft, periodic)
    ks = np.fft.rfftfreq(2 * n_s, d=DS)
    pad = np.zeros((2 * n_s, NPHI))
    pad[:n_s] = grid
    sm = np.fft.irfft(np.fft.rfft(pad, axis=0) *
                      np.exp(-2 * (np.pi * ks * SIG_S) ** 2)[:, None],
                      n=2 * n_s, axis=0)[:n_s]
    kp = np.fft.fftfreq(NPHI) * NPHI
    sig_bins = SIG_PHI / (2 * np.pi / NPHI)
    sm = np.fft.ifft(np.fft.fft(sm, axis=1) *
                     np.exp(-2 * (np.pi * kp / NPHI * sig_bins) ** 2)[None, :],
                     axis=1).real
    return sm


def correlate_tables(rhoA, rhoB, n_s, sub):
    """c[D, pa, pb] = sum_s rhoA[s, pa] rhoB[s + D, pb] via FFT over s.

    rho are (n_s, NPHI); output subsampled to (nD, NPHI_OUT, NPHI_OUT) with
    stagger step DSTEP."""
    m = 1 << int(np.ceil(np.log2(2 * n_s)))
    fa = np.fft.rfft(rhoA[:, sub], m, axis=0)          # (m/2+1, nphi_out)
    fb = np.fft.rfft(rhoB[:, sub], m, axis=0)
    # broadcast pairwise: (freq, pa, pb)
    prod = np.conj(fa)[:, :, None] * fb[:, None, :]
    c = np.fft.irfft(prod, m, axis=0)                  # lag j: sum a[i] b[i+j]
    lags_full = np.concatenate([c[m - (n_s - 1):], c[:n_s]], axis=0)
    step = int(round(DSTEP / DS))
    return lags_full[::step].astype(np.float32), (np.arange(-(n_s - 1), n_s)[::step]) * DS


def convolve_tables(rhoA, rhoB, n_s, sub):
    m = 1 << int(np.ceil(np.log2(2 * n_s)))
    fa = np.fft.rfft(rhoA[:, sub], m, axis=0)
    fb = np.fft.rfft(rhoB[:, sub], m, axis=0)
    prod = fa[:, :, None] * fb[:, None, :]
    c = np.fft.irfft(prod, m, axis=0)[:2 * n_s - 1]
    step = int(round(DSTEP / DS))
    return c[::step].astype(np.float32), np.arange(0, 2 * n_s - 1)[::step] * DS


ATELO = "--atelo" in os.sys.argv
SUFFIX = "_atelo" if ATELO else ""


def main():
    meta = json.load(open(os.path.join(ASSETS, "atoms_meta.json")))
    L, D = meta["length_nm"], meta["d_period_nm"]
    S0 = 8.0                                  # axial margin for telopeptides
    n_s = int(np.ceil((L + 2 * S0) / DS)) + 1

    surf = load_residue_surface()
    telo = [] if ATELO else telopeptides(L)
    print(f"{len(surf)} helix + {len(telo)} telopeptide residues on grid ({n_s} x {NPHI})")
    sv, phv, qv, hv = surf[:, 0] + S0, surf[:, 1], surf[:, 2], surf[:, 3]
    tv = surf[:, 4].astype(int)
    hmean = (hv.sum() + sum(h for _, _, h, _ in telo)) / (len(surf) + len(telo))
    hv = hv - hmean
    telo_h = [(s + S0, h - hmean) for s, q, h, t in telo]

    rho_h = deposit_2d(sv, phv, hv, n_s, telo=telo_h)

    # per-titratable-type unit densities (positions only; charge applied at pH)
    rho_t = []
    counts = np.zeros(NTYPES)
    for a in range(NTYPES):
        m = tv == a
        telo_a = [(s + S0, 1.0) for s, q, h, t in telo if t == a]
        counts[a] = m.sum() + len(telo_a)
        rho_t.append(deposit_2d(sv[m], phv[m], np.ones(m.sum()), n_s, telo=telo_a))
    print("titratable counts per molecule (D,E,H,K,R):", counts.astype(int).tolist())

    sub = np.linspace(0, NPHI, NPHI_OUT, endpoint=False).astype(int)

    # symmetric basis tables S_ab = C_ab + C_ba (a<b), C_aa (a=b)
    basis_par, basis_ap = [], []
    lagsD = lagsR = None
    for a in range(NTYPES):
        for b in range(a, NTYPES):
            cab, lagsD = correlate_tables(rho_t[a], rho_t[b], n_s, sub)
            if b != a:
                cba, _ = correlate_tables(rho_t[b], rho_t[a], n_s, sub)
                cab = cab + cba
            basis_par.append(cab)
            dab, lagsR = convolve_tables(rho_t[a], rho_t[b][::-1, ::-1], n_s, sub)
            if b != a:
                dba, _ = convolve_tables(rho_t[b], rho_t[a][::-1, ::-1], n_s, sub)
                dab = dab + dba
            basis_ap.append(dab)

    def charges_at(pH):
        return np.array([
            -1.0 / (1.0 + 10.0 ** (3.65 - pH)),    # Asp
            -1.0 / (1.0 + 10.0 ** (4.25 - pH)),    # Glu
            +1.0 / (1.0 + 10.0 ** (pH - 6.00)),    # His
            +1.0 / (1.0 + 10.0 ** (pH - 10.53)),   # Lys
            +1.0 / (1.0 + 10.0 ** (pH - 12.48)),   # Arg
        ])

    def combine(basis, qs):
        out = np.zeros_like(basis[0])
        k = 0
        for a in range(NTYPES):
            for b in range(a, NTYPES):
                w = qs[a] * qs[b]
                out += w * basis[k]
                k += 1
        return out

    q74 = charges_at(7.4)
    el_par = combine(basis_par, q74)
    el_ap = combine(basis_ap, q74)
    hy_par, _ = correlate_tables(rho_h, rho_h, n_s, sub)
    hy_par = -hy_par
    hy_ap, lagsR = convolve_tables(rho_h, rho_h[::-1, ::-1], n_s, sub)
    hy_ap = -hy_ap

    # calibrate: deepest parallel el well beyond |D|>5nm over all angles = -1
    mask = np.abs(lagsD) > 5.0
    el_scale = abs(min(el_par[mask].min(), -1e-9))
    hy_scale = abs(min(hy_par[mask].min(), -1e-9))
    el_par /= el_scale; el_ap /= el_scale
    hy_par /= hy_scale; hy_ap /= hy_scale
    for arr in (el_par, hy_par, el_ap, hy_ap):
        np.clip(arr, -3.0, 3.0, out=arr)

    # trimer molecular weight incl. telopeptides (for w/v concentration UI)
    mw = 18.02 * 3
    DATA = os.path.join(ROOT, "data")
    for fasta, copies in (("P02452.fasta", 2), ("P08123.fasta", 1)):
        seq = "".join(l.strip() for l in open(os.path.join(DATA, fasta))
                      if not l.startswith(">"))
        best = (0, 0)
        i = 0
        while i < len(seq):
            if seq[i] == "G":
                n = 0
                while i + 3 * n < len(seq) and seq[i + 3 * n] == "G":
                    n += 1
                if n > best[1]:
                    best = (i, n)
            i += 1
        start, ng = best
        chain = seq[max(0, start - 16):start + 3 * ng + 25]
        mw += copies * sum(MASS1.get(aa, 110.0) for aa in chain)
    mw_kda = mw / 1000.0
    print(f"trimer MW = {mw_kda:.1f} kDa")

    with open(os.path.join(ASSETS, f"profiles2d_basis{SUFFIX}.bin"), "wb") as fh:
        fh.write(b"PRFB")
        np.array([1, el_par.shape[0], NPHI_OUT, el_ap.shape[0], NTYPES],
                 dtype="<u4").tofile(fh)
        np.array([DSTEP, L, D, S0, el_scale, mw_kda], dtype="<f4").tofile(fh)
        counts.astype("<f4").tofile(fh)
        for arr in basis_par:
            (arr / el_scale).astype("<f4").tofile(fh)
        for arr in basis_ap:
            (arr / el_scale).astype("<f4").tofile(fh)
    print(f"wrote profiles2d_basis{SUFFIX}.bin")

    with open(os.path.join(ASSETS, f"profiles2d{SUFFIX}.bin"), "wb") as fh:
        fh.write(b"PRF2")
        np.array([1, el_par.shape[0], NPHI_OUT, el_ap.shape[0]], dtype="<u4").tofile(fh)
        np.array([DSTEP, L, D, S0], dtype="<f4").tofile(fh)
        for arr in (el_par, hy_par, el_ap, hy_ap):
            arr.astype("<f4").tofile(fh)
    sz = os.path.getsize(os.path.join(ASSETS, "profiles2d.bin")) / 1e6
    print(f"wrote profiles2d.bin ({sz:.1f} MB): par {el_par.shape}, ap {el_ap.shape}")

    # ---- validation: does azimuth resolution restore D contrast physically?
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comb = el_par + 0.4 * hy_par
    eff = comb.min(axis=(1, 2))          # best-over-angles landscape
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(lagsD, eff, lw=0.8, color="k")
    for k in range(1, 5):
        axes[0].axvline(k * D, color="r", ls="--", lw=0.8)
    axes[0].set_xlim(0, L)
    axes[0].set_title("min over facing angles of E(stagger) — raw, no sharpening; "
                      "red = k*D")
    i_best = np.unravel_index(np.argmin(comb), comb.shape)
    axes[1].imshow(comb[:, :, i_best[2]].T, aspect="auto", origin="lower",
                   extent=[lagsD[0], lagsD[-1], -180, 180], cmap="RdBu_r",
                   vmin=-1.2, vmax=1.2)
    axes[1].set_xlim(0, L)
    axes[1].set_title(f"E(stagger, facingA) at fixed facingB (best); blue = attractive")
    axes[1].set_ylabel("facing A (deg)")
    prof_at = comb[:, i_best[1], i_best[2]]
    axes[2].plot(lagsD, prof_at, lw=0.8, color="purple")
    for k in range(1, 5):
        axes[2].axvline(k * D, color="r", ls="--", lw=0.8)
    axes[2].set_xlim(0, L)
    axes[2].set_title("E(stagger) at the globally best facing pair")
    axes[2].set_xlabel("stagger (nm)")
    fig.tight_layout()
    fig.savefig(os.path.join(VALID, "profiles2d.png"), dpi=130)

    print("effective (best-over-angles) landscape minima:")
    local_min_report(eff, lagsD, D)
    print("done")


def local_min_report(eff, lags, D):
    idx = [i for i in range(2, len(eff) - 2)
           if eff[i] < eff[i - 1] and eff[i] < eff[i + 1] and lags[i] > 5]
    idx.sort(key=lambda i: eff[i])
    out = []
    for i in idx[:8]:
        if all(abs(lags[i] - lags[j]) > 15 for j in out):
            out.append(i)
    for i in sorted(out, key=lambda i: eff[i])[:6]:
        print(f"  eff-min at {lags[i]:7.1f} nm = {lags[i]/D:5.2f} D  depth {eff[i]:+.3f}")


if __name__ == "__main__":
    main()
