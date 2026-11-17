"""Which residues touch at contact, against what the coarse hydrophobic term weights.

``python -m aamd.contacts SYSTEM WINDOW [WINDOW ...]``  ->  aamd/results/contacts_SYSTEM.json

For the frames of the given windows: every residue pair (one residue of each
segment) with any two heavy atoms within 0.45 nm is "in contact" in that
frame. Reported:

* contact frequency per residue pair, and per residue class (apolar: A V L I
  M F; imino: P, Hyp; small polar: G S T N Q Y; charged: D E K R H and the
  acyl-lysines);
* the side-chain atom classes in contact (carbon-carbon against
  polar-polar heavy atoms), the crude measure of how much of the interface is
  hydrophobic;
* the Spearman rank correlation, over residue pairs, between contact
  frequency and the coarse model's hydrophobic weight product h_i h_j
  (centred Kyte-Doolittle), and between contact frequency and -h_i h_j
  restricted to pairs the coarse model calls attractive.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

from modcoll.sites import KYTE_DOOLITTLE, native_molecule

from . import build

CLASSES = {"apolar": set("AVLIMF"), "imino": set("PO"), "polar": set("GSTNQYW"), "charged": set("DEKRH")}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLH": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H", "ILE": "I", "LEU": "L",
       "LYS": "K", "LYN": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "HYP": "O", "SLL": "K", "SLH": "K", "ALY": "K"}


def residue_class(one):
    return next(k for k, v in CLASSES.items() if one in v)


def analyse(name, windows, cutoff=0.45, stride=2):
    import mdtraj

    sysdir = build.WORK / name
    mol = native_molecule()
    h_mean = np.mean([KYTE_DOOLITTLE[a] for a in mol.aa[mol.is_residue]])
    pair_freq = {}
    atom_cls = {"C-C": 0, "C-polar": 0, "polar-polar": 0}
    n_frames = 0
    for w in windows:
        jd = sysdir / f"w{w:02d}"
        dcds = sorted(glob.glob(str(jd / "solute_*.dcd")))
        if not dcds:
            continue
        t = mdtraj.load(dcds, top=str(jd / "solute.pdb"))[::stride]
        top = t.topology
        prot = [r for r in top.residues if r.name in AA3 or r.name in ("ACE", "NME")]
        per = 3 * 32
        seg = {r.index: (0 if k < per else 1) for k, r in enumerate(prot)}
        heavy = [a.index for a in top.atoms if a.element.symbol != "H" and a.residue.index in seg
                 and a.residue.name in AA3]
        A = np.array([i for i in heavy if seg[top.atom(i).residue.index] == 0])
        B = np.array([i for i in heavy if seg[top.atom(i).residue.index] == 1])
        resA = np.array([top.atom(i).residue.index for i in A])
        resB = np.array([top.atom(i).residue.index for i in B])
        polarA = np.array([top.atom(i).element.symbol in "NO" for i in A])
        polarB = np.array([top.atom(i).element.symbol in "NO" for i in B])
        for f in range(t.n_frames):
            x = t.xyz[f]
            d = np.linalg.norm(x[A][:, None] - x[B][None], axis=-1)
            ia, ib = np.nonzero(d < cutoff)
            seen = set(zip(resA[ia], resB[ib]))
            for p in seen:
                pair_freq[p] = pair_freq.get(p, 0) + 1
            for a, b in zip(ia, ib):
                k = "polar-polar" if polarA[a] and polarB[b] else ("C-C" if not polarA[a] and not polarB[b]
                                                                   else "C-polar")
                atom_cls[k] += 1
            n_frames += 1
        names = {r.index: (r.name, AA3[r.name]) for r in prot if r.name in AA3}
    if n_frames == 0:
        return None
    rows = []
    for (ra, rb), n in pair_freq.items():
        a1, b1 = names[ra][1], names[rb][1]
        h = (KYTE_DOOLITTLE[a1] - h_mean) * (KYTE_DOOLITTLE[b1] - h_mean)
        rows.append({"res_a": names[ra][0], "res_b": names[rb][0], "freq": n / n_frames, "h_product": h,
                     "classes": "-".join(sorted((residue_class(a1), residue_class(b1))))})
    by_cls = {}
    for r in rows:
        by_cls[r["classes"]] = by_cls.get(r["classes"], 0.0) + r["freq"]
    tot = sum(by_cls.values())
    from scipy.stats import spearmanr

    f = np.array([r["freq"] for r in rows])
    hp = np.array([r["h_product"] for r in rows])
    rho_all = spearmanr(f, hp).correlation if len(rows) > 3 else float("nan")
    att = hp > 0
    rho_att = spearmanr(f[att], hp[att]).correlation if att.sum() > 3 else float("nan")
    n_at = sum(atom_cls.values())
    out = {"system": name, "windows": list(windows), "frames": n_frames,
           "contacts_per_frame": float(sum(pair_freq.values()) / n_frames),
           "residue_pairs_seen": len(rows),
           "share_by_class": {k: v / tot for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1])},
           "atom_contacts_share": {k: v / n_at for k, v in atom_cls.items()} if n_at else {},
           "spearman_freq_vs_h_product": float(rho_all), "spearman_freq_vs_h_product_attractive": float(rho_att),
           "share_of_contacts_on_pairs_coarse_calls_attractive": float(f[att].sum() / f.sum()) if f.sum() else None,
           "top_pairs": sorted(rows, key=lambda r: -r["freq"])[:15]}
    (build.AAMD / "results").mkdir(exist_ok=True)
    (build.AAMD / "results" / f"contacts_{name}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    o = analyse(a[0], [int(x) for x in a[1:]])
    print(json.dumps({k: v for k, v in o.items() if k != "top_pairs"}, indent=1))


if __name__ == "__main__":
    main()
