"""
Force-matching harness: all-atom implicit-solvent (GBSA/OBC2) mean-force
measurements between tropocollagen SEGMENT PAIRS, for Delta-learning
corrections to the registry tables consumed by the BD / MC / KMC engines.

Method: excise two 4 nm triple-helix segments from the atomistic template at
contour positions matching a molecular stagger Delta (B's window = A's window
shifted by -Delta), pose them side by side (gap, twist angles), restrain the
backbone harmonically, run Langevin MD at 310 K, and read the mean
inter-segment force from the restraint displacements:
    <F_inter> = k * sum_over_B(<x - x0>)
(with backbone restrained and internal forces summing to zero, the restraint
offset balances exactly the inter-segment force).

HYP is mutated to PRO for amber14 compatibility (v1; the hydroxyl's hydration
role is partly folded into GBSA — revisit with CHARMM36/explicit water later).

Usage:
  python tools/pmf_harness.py smoke                 # one pose, quick check
  python tools/pmf_harness.py poses                 # write poses.json schedule
  python tools/pmf_harness.py run [--ns 0.3]        # batch with resume (CSV)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "pmf_data")
os.makedirs(OUT, exist_ok=True)

SEG_NM = 4.0          # segment length
KREST = 2000.0        # kJ/mol/nm^2 backbone restraint
KT_KJ = 2.577         # kJ/mol per kT at 310 K


def load_template():
    """residues[(chain, resnum)] = {'name': rn, 'atoms': [(aname, elem, xyz_A)]}"""
    res = {}
    path = os.path.join(ASSETS, "tropocollagen.pdb")
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            aname = line[12:16].strip()
            rn = line[17:20].strip()
            cid = line[21]
            ri = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            e = line[76:78].strip()
            d = res.setdefault((cid, ri), {"name": rn, "atoms": []})
            d["atoms"].append((aname, e, xyz))
    return res


def excise(res, s0_nm, alpha, out_lines, chain_offset):
    """Write one segment (3 chains) rotated by alpha about its own axis and
    centered at origin; returns count of residues written."""
    z0, z1 = s0_nm * 10.0, (s0_nm + SEG_NM) * 10.0
    picked = []
    for (cid, ri), d in res.items():
        ca = next((a for a in d["atoms"] if a[0] == "CA"), None)
        if ca and z0 <= ca[2][2] < z1:
            picked.append((cid, ri, d))
    if not picked:
        raise ValueError(f"no residues in window {s0_nm}")
    zs = [a[2][2] for _, _, d in picked for a in d["atoms"]]
    zc = 0.5 * (min(zs) + max(zs))
    ca_, sa_ = np.cos(alpha), np.sin(alpha)
    serial = [1]
    nres = 0
    for cid in "ABC":
        rs = sorted([(ri, d) for (c, ri, d) in picked if c == cid])
        for newri, (ri, d) in enumerate(rs, start=1):
            rn = d["name"]
            drop_od1 = rn == "HYP"
            rn_out = "PRO" if rn == "HYP" else rn
            for aname, e, (x, y, z) in d["atoms"]:
                if drop_od1 and aname == "OD1":
                    continue
                xr = ca_ * x - sa_ * y
                yr = sa_ * x + ca_ * y
                zr = z - zc
                out_lines.append(
                    f"ATOM  {serial[0] % 100000:5d} {aname:<4s} {rn_out:>3s} "
                    f"{chr(ord(chain_offset) + 'ABC'.index(cid))}{newri:4d}    "
                    f"{xr:8.3f}{yr:8.3f}{zr:8.3f}  1.00  0.00          {e:>2s}\n")
                serial[0] += 1
            nres += 1
        out_lines.append("TER\n")
    return nres


def build_pose_pdb(res, sA, delta, gap, alpha, beta, path, tilt=0.0):
    """Segment A at origin; segment B (contour window sA - delta) at +x gap,
    optionally tilted about the gap (x) axis so contact stays at the centers."""
    lines = []
    excise(res, sA, alpha, lines, "A")
    linesB = []
    excise(res, sA - delta, beta, linesB, "D")
    ct, st = np.cos(tilt), np.sin(tilt)
    for ln in linesB:
        if ln.startswith("ATOM"):
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
            yr = ct * y - st * z
            zr = st * y + ct * z
            ln = (ln[:30] + f"{x + gap * 10.0:8.3f}{yr:8.3f}{zr:8.3f}" + ln[54:])
        lines.append(ln)
    with open(path, "w") as fh:
        fh.writelines(lines)
        fh.write("END\n")


def run_pose(res, sA, delta, gap, alpha, beta, ns=0.3, tilt=0.0,
             platform_pref="OpenCL"):
    from openmm import app, unit, LangevinMiddleIntegrator, CustomExternalForce, Platform
    from pdbfixer import PDBFixer

    tmp = os.path.join(OUT, "pose_tmp.pdb")
    build_pose_pdb(res, sA, delta, gap, alpha, beta, tmp, tilt=tilt)
    fixer = PDBFixer(filename=tmp)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds)

    # backbone restraints; tag which atoms belong to segment B (chains D,E,F)
    rest = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", KREST)
    for par in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(par)
    pos0 = fixer.positions
    bIdx = []
    x0 = {}
    for atom in fixer.topology.atoms():
        if atom.name in ("N", "CA", "C"):
            p = pos0[atom.index].value_in_unit(unit.nanometer)
            rest.addParticle(atom.index, [p[0], p[1], p[2]])
            x0[atom.index] = np.array(p)
            if atom.residue.chain.id in "DEF":
                bIdx.append(atom.index)
    system.addForce(rest)

    integ = LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond,
                                     0.002 * unit.picoseconds)
    try:
        plat = Platform.getPlatformByName(platform_pref)
    except Exception:
        plat = Platform.getPlatformByName("CPU")
    simu = app.Simulation(fixer.topology, system, integ, plat)
    simu.context.setPositions(pos0)
    simu.minimizeEnergy(maxIterations=500)
    simu.step(10000)                       # 20 ps equilibration

    nChunks = max(10, int(ns * 1000))
    acc = np.zeros(3)
    accT = np.zeros(3)
    cB = np.mean([x0[i] for i in bIdx], axis=0)
    nS = 0
    for _ in range(nChunks):
        simu.step(500)                     # 1 ps between samples
        st = simu.context.getState(getPositions=True)
        pns = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        for i in bIdx:
            d = pns[i] - x0[i]
            acc += d
            accT += np.cross(x0[i] - cB, d)
        nS += 1
    # mean force / torque on B from restraint offsets, kJ/mol -> kT units
    F = KREST * (acc / nS) / KT_KJ
    T = KREST * (accT / nS) / KT_KJ       # kT per radian, about B's center
    return F, T, plat.getName()


def gen_poses():
    D = 66.88
    poses = []
    deltas = set()
    # both windows must fit: 0 <= sA - delta and sA + SEG <= L
    sa_by_k = {1: (80.0, 130.0, 180.0, 230.0),
               2: (140.0, 180.0, 230.0, 270.0),
               3: (210.0, 240.0, 265.0, 285.0)}
    for k in (1, 2, 3):
        for ddz in (-3.0, -1.5, 0.0, 1.5, 3.0):
            for sA in sa_by_k[k]:
                if sA - (k * D + ddz) < 0.5 or sA + SEG_NM > 290.0:
                    continue
                for ab in ((0, 0), (1.57, 1.57), (3.14, 0), (0, 3.14)):
                    poses.append({"sA": sA, "delta": k * D + ddz, "gap": 1.5,
                                  "alpha": ab[0], "beta": ab[1]})
                deltas.add((sA, round(k * D + ddz, 2)))
    # off-well background controls + the predicted zero-stagger repulsion zone
    for dz in (0.5, 20.0, 33.4, 50.2):
        for sA in (80.0, 180.0):
            for ab in ((0, 0), (1.57, 1.57)):
                poses.append({"sA": sA, "delta": dz, "gap": 1.5,
                              "alpha": ab[0], "beta": ab[1]})
            deltas.add((sA, round(dz, 2)))
    for g in (1.2, 1.8, 2.2):              # radial scan at the 1D well
        for sA in (80.0, 180.0):
            poses.append({"sA": sA, "delta": D, "gap": g, "alpha": 0, "beta": 0})
    # far-gap baselines: cut-terminus charge artifact + GBSA tail, subtracted
    # per (sA, delta) at merge time
    for sA, dz in sorted(deltas):
        poses.append({"sA": sA, "delta": dz, "gap": 6.0, "alpha": 0, "beta": 0})
    with open(os.path.join(OUT, "poses.json"), "w") as fh:
        json.dump(poses, fh, indent=1)
    print(f"wrote {len(poses)} poses to pmf_data/poses.json "
          f"(incl. {len(deltas)} far-gap baselines)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    res = load_template()
    if cmd == "poses":
        gen_poses()
    elif cmd == "smoke":
        import time
        t0 = time.time()
        F, _, plat = run_pose(res, 130.0, 66.88, 1.5, 0.0, 0.0, ns=0.05)
        print(f"platform {plat} | pose (sA=130, D-stagger, gap 1.5): "
              f"F = ({F[0]:+.2f}, {F[1]:+.2f}, {F[2]:+.2f}) kT/nm "
              f"| {time.time()-t0:.0f} s")
        print("Fx < 0 means attraction (B pulled toward A at -x): "
              f"{'ATTRACTIVE' if F[0] < 0 else 'repulsive'}")
    elif cmd == "run":
        ns = 0.3
        if "--ns" in sys.argv:
            ns = float(sys.argv[sys.argv.index("--ns") + 1])
        poses = json.load(open(os.path.join(OUT, "poses.json")))
        csv = os.path.join(OUT, "forces.csv")
        done = set()
        if os.path.exists(csv):
            for ln in open(csv).readlines()[1:]:
                t = ln.split(",")
                done.add((float(t[0]), float(t[1]), float(t[2]),
                          float(t[3]), float(t[4])))
        else:
            with open(csv, "w") as fh:
                fh.write("sA,delta,gap,alpha,beta,Fx,Fy,Fz,ns\n")
        for i, ps in enumerate(poses):
            key = (ps["sA"], ps["delta"], ps["gap"], ps["alpha"], ps["beta"])
            if key in done:
                continue
            try:
                F, _, _ = run_pose(res, *key, ns=ns)
            except Exception as ex:
                print(f"[{i+1}/{len(poses)}] SKIP {key}: {ex}")
                continue
            with open(csv, "a") as fh:
                fh.write(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},"
                         f"{F[0]:.4f},{F[1]:.4f},{F[2]:.4f},{ns}\n")
            print(f"[{i+1}/{len(poses)}] delta={ps['delta']:.1f} gap={ps['gap']} "
                  f"-> Fz {F[2]:+.2f} Fx {F[0]:+.2f} kT/nm")
            sys.stdout.flush()
    elif cmd == "tilt":
        # angular series: calibrate the tilt falloff of the registry
        # interaction and measure the alignment torque; own CSV, resumable
        ns = 1.0
        if "--ns" in sys.argv:
            ns = float(sys.argv[sys.argv.index("--ns") + 1])
        D = 66.88
        poses = []
        for dz in (D, 2 * D):
            for sA in ((80.0, 180.0) if dz < 100 else (180.0, 230.0)):
                for ab in ((0.0, 0.0), (1.57, 1.57)):
                    for tl in (0.0, 0.087, 0.175, 0.349, 0.785, 1.571):
                        for gp in (1.5, 6.0):     # contact + baseline
                            poses.append((sA, round(dz, 2), gp, ab[0], ab[1],
                                          round(tl, 3)))
        csv = os.path.join(OUT, "forces_tilt.csv")
        done = set()
        if os.path.exists(csv):
            for ln in open(csv).readlines()[1:]:
                t = ln.split(",")
                done.add(tuple(float(x) for x in t[:6]))
        else:
            with open(csv, "w") as fh:
                fh.write("sA,delta,gap,alpha,beta,tilt,Fx,Fy,Fz,Tx,ns\n")
        for i, key in enumerate(poses):
            if key in done:
                continue
            try:
                F, T, _ = run_pose(res, key[0], key[1], key[2], key[3], key[4],
                                   ns=ns, tilt=key[5])
            except Exception as ex:
                print(f"[{i+1}/{len(poses)}] SKIP {key}: {ex}")
                continue
            with open(csv, "a") as fh:
                fh.write(",".join(str(k) for k in key) +
                         f",{F[0]:.4f},{F[1]:.4f},{F[2]:.4f},{T[0]:.4f},{ns}\n")
            print(f"[{i+1}/{len(poses)}] d={key[1]:.1f} tilt={key[5]:.2f} "
                  f"gap={key[2]} -> Fx {F[0]:+.2f} Fz {F[2]:+.2f} Tx {T[0]:+.2f}")
            sys.stdout.flush()
    elif cmd == "tiltdeep":
        # depth over breadth: replicate the strongest-attraction contexts at
        # fine angles for a statistically resolvable falloff curve
        import csv as csvmod
        rows = list(csvmod.DictReader(open(os.path.join(OUT, "forces_tilt.csv"))))
        ctx = {}
        for r in rows:
            key = (float(r["sA"]), float(r["delta"]), float(r["alpha"]),
                   float(r["beta"]), float(r["tilt"]))
            ctx.setdefault(key[:4], {}).setdefault(key[4], {})[float(r["gap"])] = r
        ranked = []
        for c, tilts in ctx.items():
            g = tilts.get(0.0, {})
            if 1.5 in g and 6.0 in g:
                f0 = float(g[1.5]["Fx"]) - float(g[6.0]["Fx"])
                if f0 < -5.0:            # strong attraction only
                    ranked.append((f0, c))
        ranked.sort()
        picks = [c for _, c in ranked[:2]]
        print("deep contexts:", picks)
        out = os.path.join(OUT, "forces_tiltdeep.csv")
        done = set()
        if os.path.exists(out):
            for ln in open(out).readlines()[1:]:
                t = ln.split(",")
                done.add((float(t[0]), float(t[1]), float(t[2]), float(t[3]),
                          float(t[4]), float(t[5]), int(t[6])))
        else:
            with open(out, "w") as fh:
                fh.write("sA,delta,gap,alpha,beta,tilt,rep,Fx,Fy,Fz,Tx,ns\n")
        jobs = []
        for c in picks:
            for tl in (0.0, 0.087, 0.175, 0.262, 0.349, 0.524):
                for rep in range(3):
                    jobs.append((c[0], c[1], 1.5, c[2], c[3], round(tl, 3), rep))
                if tl not in (0.0, 0.087, 0.175, 0.349):   # new baselines
                    jobs.append((c[0], c[1], 6.0, c[2], c[3], round(tl, 3), 0))
        for i, key in enumerate(jobs):
            if key in done:
                continue
            try:
                F, T, _ = run_pose(res, key[0], key[1], key[2], key[3], key[4],
                                   ns=2.0, tilt=key[5])
            except Exception as ex:
                print(f"[{i+1}/{len(jobs)}] SKIP {key}: {ex}")
                continue
            with open(out, "a") as fh:
                fh.write(",".join(str(k) for k in key) +
                         f",{F[0]:.4f},{F[1]:.4f},{F[2]:.4f},{T[0]:.4f},2.0\n")
            print(f"[{i+1}/{len(jobs)}] tilt={key[5]:.2f} rep={key[6]} gap={key[2]} "
                  f"-> Fx {F[0]:+.2f} Tx {T[0]:+.2f}")
            sys.stdout.flush()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
