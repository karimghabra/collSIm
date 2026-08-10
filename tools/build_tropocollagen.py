"""
Build a full-atom (heavy atom) model of human type I tropocollagen.

Method:
  1. Parse 1K6F [(Pro-Pro-Gly)10]3 at 1.30 A. Chains A,B,C form one triple helix.
  2. Extract the triple-helix screw operator (rotation theta + rise h per triplet)
     by least-squares superposition of triplet t onto triplet t+1 across all
     three chains simultaneously, averaged over the middle of the helix.
  3. Canonicalize: screw axis -> +Z. Keep the middle triplet of each chain as the
     backbone template. An infinitely extendable ideal triple helix is then
     residue(slot, k) = Rz(k*theta) * x_slot + k*h*Z.
  4. Thread the real helical-domain sequences (COL1A1 x2 -> chains A,B; COL1A2 -> C,
     assignment of the heterotrimer permutation is arbitrary). Y-position prolines
     are hydroxylated (HYP), the dominant modification in vivo.
  5. Side chains: superpose PDB Chemical Component Dictionary ideal residues onto
     each placed backbone via (N,CA,C); keep template backbone N,CA,C,O; take CCD
     atoms beyond the backbone. Chi1 is relaxed greedily over 6 rotations about
     CA-CB against already-placed atoms.

Outputs (assets/):
  atoms.bin            packed per-atom records for the GPU renderer
  tropocollagen.pdb    full model for external inspection
  atoms_meta.json      parameters + validation numbers
  residues.npz         per-residue data for the interaction-profile step
  validation/template_views.png
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ASSETS = os.path.join(ROOT, "assets")
VALID = os.path.join(ASSETS, "validation")
os.makedirs(VALID, exist_ok=True)

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
# color classes: 0 nonpolar, 1 polar, 2 positive, 3 negative, 4 glycine
def color_class(aa1):
    if aa1 == "G":
        return 4
    if aa1 in "KRH":
        return 2
    if aa1 in "DE":
        return 3
    if aa1 in "STNQYCW":
        return 1
    return 0

ELEMENTS = {"C": 0, "N": 1, "O": 2, "S": 3}
BACKBONE = ("N", "CA", "C", "O")


# ---------------------------------------------------------------- geometry ---
def kabsch(P, Q):
    """Return R, t minimizing ||R@P.T + t - Q.T|| (rows are points, P->Q)."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def rotation_axis_angle(R):
    """Unit axis and signed angle of rotation matrix."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(w) / 2.0
    c = (np.trace(R) - 1.0) / 2.0
    angle = np.arctan2(s, c)
    if s < 1e-9:
        raise ValueError("degenerate rotation")
    axis = w / (2.0 * s)
    return axis, angle


def rot_to(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------- 1K6F parsing ---
def parse_1k6f(path):
    """chains[cid] = list of {name: xyz} per residue ordinal, plus residue names."""
    chains = defaultdict(dict)   # cid -> resSeq -> {atom: xyz}
    names = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            cid = line[21]
            if cid not in "ABC":
                continue
            name = line[12:16].strip()
            res = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains[cid].setdefault(res, {})[name] = xyz
            names[cid][res] = line[17:20].strip()
    out = {}
    for cid in "ABC":
        keys = sorted(chains[cid])
        out[cid] = [chains[cid][k] for k in keys]
        rn = [names[cid][k] for k in keys]
        assert len(out[cid]) >= 27, f"chain {cid}: only {len(out[cid])} residues"
        assert all(n == "GLY" for n in rn[2::3]), "expected GLY at every 3rd position"
    return out


def extract_screw(chains):
    """Least-squares screw operator mapping triplet t -> t+1 (all chains at once)."""
    Ps, Qs = [], []
    for t in range(2, 7):          # middle triplets, avoid frayed ends
        for cid in "ABC":
            for r in range(3 * t, 3 * t + 3):
                for a in BACKBONE:
                    Ps.append(chains[cid][r][a])
                    Qs.append(chains[cid][r + 3][a])
    R, tvec = kabsch(np.array(Ps), np.array(Qs))
    axis, angle = rotation_axis_angle(R)
    rise = float(tvec @ axis)
    if rise < 0:                   # orient +axis along increasing residue index
        axis, angle, rise = -axis, -angle, -rise
    # point on screw axis: (I-R)p = t_perp
    t_perp = tvec - rise * axis
    p0, *_ = np.linalg.lstsq(np.eye(3) - R, t_perp, rcond=None)
    rms = np.sqrt(np.mean(np.sum((np.array(Ps) @ R.T + tvec - np.array(Qs)) ** 2, axis=1)))
    return R, tvec, axis, angle, rise, p0, rms


# --------------------------------------------------------------- CCD ideal ---
def parse_ccd(code):
    """Heavy atoms (minus OXT) with ideal coordinates from a CCD cif file."""
    atoms = {}
    order = []
    in_loop = False
    with open(os.path.join(DATA, "ccd", f"{code}.cif")) as fh:
        for line in fh:
            if line.startswith("_chem_comp_atom.pdbx_ordinal"):
                in_loop = True
                continue
            if in_loop:
                if line.startswith("#") or line.startswith("loop_"):
                    break
                tok = line.split()
                if len(tok) < 21:
                    continue
                name, elem = tok[1], tok[3]
                if elem == "H" or name == "OXT":
                    continue
                xyz = np.array([float(tok[15]), float(tok[16]), float(tok[17])])
                atoms[name] = (elem, xyz)
                order.append(name)
    if not atoms:
        raise ValueError(f"no atoms parsed for {code}")
    return atoms, order


# ------------------------------------------------------------------ threads ---
def helical_domain(fasta_path):
    seq = "".join(l.strip() for l in open(fasta_path) if not l.startswith(">"))
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
        else:
            i += 1
    start, ng = best
    end = min(start + 3 * ng, len(seq))
    dom = seq[start:end]
    return dom, start


class Grid:
    """Uniform spatial hash for clash checks."""
    def __init__(self, cell=3.0):
        self.cell = cell
        self.d = defaultdict(list)

    def key(self, p):
        return (int(np.floor(p[0] / self.cell)),
                int(np.floor(p[1] / self.cell)),
                int(np.floor(p[2] / self.cell)))

    def add(self, pts, tag):
        for p in pts:
            self.d[self.key(p)].append((p, tag))

    def near(self, p, skip_tags):
        kx, ky, kz = self.key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q, tag in self.d.get((kx + dx, ky + dy, kz + dz), ()):
                        if tag not in skip_tags:
                            yield q


def clash_score(pts, grid, skip_tags, cutoff=2.9):
    s = 0.0
    for p in pts:
        for q in grid.near(p, skip_tags):
            d = np.linalg.norm(p - q)
            if d < cutoff:
                s += (cutoff - d) ** 2
    return s


def main():
    chains = parse_1k6f(os.path.join(DATA, "1K6F.pdb"))
    R, tvec, axis, angle, rise, p0, screw_rms = extract_screw(chains)
    print(f"screw: theta = {np.degrees(angle):+.3f} deg/triplet, rise = {rise:.4f} A, fit rms = {screw_rms:.3f} A")

    # canonical frame: p0 -> origin, axis -> +Z
    Q = rot_to(axis, np.array([0.0, 0.0, 1.0]))
    def canon(x):
        return Q @ (x - p0)

    # template = middle triplet (residues 12,13,14 zero-based = 13,14,15) per chain
    # slots: index 0 -> X (Pro), 1 -> Y (Pro), 2 -> G (Gly); G slot is residue 14 (0-based)
    T0 = 12
    template = {}
    for ci, cid in enumerate("ABC"):
        template[ci] = []
        for slot in range(3):
            res = {a: canon(v) for a, v in chains[cid][T0 + slot].items() if a in BACKBONE}
            template[ci].append(res)

    theta = angle
    h = rise

    dom1, off1 = helical_domain(os.path.join(DATA, "P02452.fasta"))
    dom2, off2 = helical_domain(os.path.join(DATA, "P08123.fasta"))
    print(f"alpha1 helical domain: {len(dom1)} aa (offset {off1});  alpha2: {len(dom2)} aa (offset {off2})")
    n = min(len(dom1), len(dom2))
    n -= n % 3
    dom1, dom2 = dom1[:n], dom2[:n]
    seqs = [dom1, dom1, dom2]        # chains A,B <- alpha1; C <- alpha2

    ccd = {code: parse_ccd(code) for code in list(AA3.values()) + ["HYP"]}

    grid = Grid(cell=3.0)
    placed = []      # (chain, resIdx, resName, atomName, elem, xyz)
    rot_options = [0.0, 120.0, -120.0, 60.0, -60.0, 180.0]
    total_clash_pairs = 0

    for i in range(n):                     # residue index within each chain
        j, m = divmod(i, 3)
        # sequence residue i maps to: G (m==0) -> slot 2 of screw k=j;
        # X (m==1) -> slot 0 of k=j+1;  Y (m==2) -> slot 1 of k=j+1
        slot = (2, 0, 1)[m]
        k = j if m == 0 else j + 1
        Rk = rz(theta * k)
        dz = np.array([0.0, 0.0, h * k])
        for ci in range(3):
            aa1 = seqs[ci][i]
            resname = AA3[aa1]
            if aa1 == "P" and m == 2:       # hydroxylate Y-position prolines
                resname = "HYP"
            bb = {a: Rk @ v + dz for a, v in template[ci][slot].items()}
            atoms_out = [(a, a[0], bb[a]) for a in BACKBONE]

            if resname != "GLY":
                ideal, _ = parse_ccd_cached(ccd, resname)
                P = np.array([ideal[a][1] for a in ("N", "CA", "C")])
                Qm = np.array([bb[a] for a in ("N", "CA", "C")])
                Rf, tf = kabsch(P, Qm)
                side = [(a, e, Rf @ v + tf) for a, (e, v) in ideal.items()
                        if a not in BACKBONE]
                # chi1 relaxation about CA-CB for residues with atoms beyond CB
                if len(side) > 1 and "CB" in ideal and resname not in ("PRO", "HYP"):
                    ca = bb["CA"]
                    cb = Rf @ ideal["CB"][1] + tf
                    u = cb - ca
                    u /= np.linalg.norm(u)
                    beyond = [(a, e, v) for a, e, v in side if a != "CB"]
                    best = None
                    skip = {(ci, i), (ci, i - 1), (ci, i + 1)}
                    for deg in rot_options:
                        ang = np.radians(deg)
                        c, s = np.cos(ang), np.sin(ang)
                        pts = []
                        for a, e, v in beyond:
                            w = v - ca
                            wpar = (w @ u) * u
                            wperp = w - wpar
                            wrot = wperp * c + np.cross(u, wperp) * s + wpar
                            pts.append(wrot + ca)
                        sc = clash_score(pts, grid, skip)
                        if best is None or sc < best[0]:
                            best = (sc, pts)
                        if sc == 0.0:
                            break
                    side = [(a, e, v) for (a, e, _), v in
                            zip(beyond, best[1])] + [("CB", "C", cb)]
                    total_clash_pairs += 1 if best[0] > 0 else 0
                atoms_out += [(a, e, v) for a, e, v in side]

            tag = (ci, i)
            grid.add([v for _, _, v in atoms_out], tag)
            for a, e, v in atoms_out:
                placed.append((ci, i, resname, a, e, v))

    print(f"placed {len(placed)} heavy atoms; residues with residual chi1 clash: {total_clash_pairs}")

    # ------------------------------------------------------------ validation
    # peptide bond C(i)->N(i+1) lengths per chain
    bb_by = {}
    for ci, i, rn, a, e, v in placed:
        if a in ("N", "C", "CA"):
            bb_by[(ci, i, a)] = v
    dists = []
    for ci in range(3):
        for i in range(n - 1):
            if (ci, i, "C") in bb_by and (ci, i + 1, "N") in bb_by:
                dists.append(np.linalg.norm(bb_by[(ci, i, "C")] - bb_by[(ci, i + 1, "N")]))
    dists = np.array(dists)
    print(f"peptide C-N: mean {dists.mean():.3f} A, sd {dists.std():.3f}, min {dists.min():.3f}, max {dists.max():.3f}")
    assert abs(dists.mean() - 1.33) < 0.1 and dists.std() < 0.1, "peptide bond geometry broken - mapping error"

    # glycine core check: mean radial distance of CA by slot
    r_gly = np.mean([np.linalg.norm(v[:2]) for ci, i, rn, a, e, v in placed
                     if a == "CA" and rn == "GLY"])
    r_oth = np.mean([np.linalg.norm(v[:2]) for ci, i, rn, a, e, v in placed
                     if a == "CA" and rn != "GLY"])
    print(f"CA radial distance: Gly {r_gly:.2f} A vs non-Gly {r_oth:.2f} A (Gly must be inner)")
    assert r_gly < r_oth, "Gly not packed at core - phase error"

    zs = np.array([v[2] for _, _, _, a, _, v in placed])
    length_nm = (zs.max() - zs.min()) / 10.0
    d_period_nm = 234 * (h / 3.0) / 10.0
    print(f"molecule length {length_nm:.1f} nm, D-period (234 res) {d_period_nm:.2f} nm, L/D = {length_nm/d_period_nm:.3f}")

    # ------------------------------------------------------------- outputs
    z0 = zs.min()
    natoms = len(placed)
    rec = np.zeros(natoms, dtype=[("s", "f4"), ("x", "f4"), ("y", "f4"),
                                  ("elem", "u1"), ("cls", "u1"),
                                  ("res", "u2"), ("chain", "u1"), ("pad", "u1"),
                                  ("pad2", "u2")])
    aa1_of = {}
    for k1, v1 in AA3.items():
        aa1_of[v1] = k1
    aa1_of["HYP"] = "P"
    for idx, (ci, i, rn, a, e, v) in enumerate(placed):
        rec[idx] = ((v[2] - z0) / 10.0, v[0] / 10.0, v[1] / 10.0,
                    ELEMENTS.get(e, 0), color_class(aa1_of[rn]), i, ci, 0, 0)
    with open(os.path.join(ASSETS, "atoms.bin"), "wb") as fh:
        fh.write(b"TROP")
        np.array([1, natoms], dtype="<u4").tofile(fh)
        np.array([length_nm, d_period_nm], dtype="<f4").tofile(fh)
        np.array([n, 0, 0], dtype="<u4").tofile(fh)
        rec.tofile(fh)
    print(f"wrote atoms.bin ({natoms} atoms, {os.path.getsize(os.path.join(ASSETS,'atoms.bin'))/1024:.0f} KB)")

    # per-residue arrays for interaction profiles (axial position from CA)
    res_z = np.zeros((3, n), dtype=np.float32)
    res_q = np.zeros((3, n), dtype=np.float32)
    res_h = np.zeros((3, n), dtype=np.float32)
    for ci in range(3):
        for i in range(n):
            res_z[ci, i] = (bb_by[(ci, i, "CA")][2] - z0) / 10.0
            aa = seqs[ci][i]
            res_q[ci, i] = CHARGE.get(aa, 0.0)
            res_h[ci, i] = KYTE_DOOLITTLE[aa]
    np.savez(os.path.join(ASSETS, "residues.npz"), z=res_z, q=res_q, h=res_h,
             seq1=np.frombuffer(dom1.encode(), dtype="u1"),
             seq2=np.frombuffer(dom2.encode(), dtype="u1"))

    # full PDB (for inspection in PyMOL/ChimeraX)
    with open(os.path.join(ASSETS, "tropocollagen.pdb"), "w") as fh:
        serial = 1
        for ci, i, rn, a, e, v in placed:
            cid = "ABC"[ci]
            fh.write(f"ATOM  {serial % 100000:5d} {a:<4s} {rn:>3s} {cid}{(i + 1) % 10000:4d}    "
                     f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}  1.00  0.00          {e:>2s}\n")
            serial += 1
        fh.write("END\n")

    meta = {
        "theta_deg_per_triplet": float(np.degrees(theta)),
        "rise_A_per_triplet": float(h),
        "screw_fit_rms_A": float(screw_rms),
        "n_res_per_chain": int(n),
        "n_atoms": int(natoms),
        "length_nm": float(length_nm),
        "d_period_nm": float(d_period_nm),
        "peptide_CN_mean_A": float(dists.mean()),
        "peptide_CN_sd_A": float(dists.std()),
        "gly_CA_radius_A": float(r_gly),
        "nongly_CA_radius_A": float(r_oth),
        "chains": ["alpha1(I)", "alpha1(I)", "alpha2(I)"],
        "hydroxyproline": "all Y-position Pro",
        "telopeptides": "excluded (helical domain only)",
    }
    with open(os.path.join(ASSETS, "atoms_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    render_validation(rec, length_nm)
    print("done")


def parse_ccd_cached(ccd, resname):
    return ccd[resname]


def render_validation(rec, length_nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[2, 1])
    seg = rec[(rec["s"] > 10) & (rec["s"] < 25)]
    cols = {0: "#888", 1: "#3a7", 2: "#26c", 3: "#c22", 4: "#fa0"}
    for cls in cols:
        m = seg[seg["cls"] == cls]
        axes[0].scatter(m["s"], m["x"], s=1.5, c=cols[cls], lw=0)
    axes[0].set_title("side view, 15 nm segment (colored by residue class; orange = Gly core)")
    axes[0].set_xlabel("axial s (nm)"); axes[0].set_ylabel("x (nm)")
    axes[0].set_aspect("equal")
    axes[1].scatter(seg["x"], seg["y"], s=1.5,
                    c=[cols[c] for c in seg["cls"]], lw=0)
    axes[1].set_title("axial projection (cross-section)")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(VALID, "template_views.png"), dpi=130)


if __name__ == "__main__":
    main()
