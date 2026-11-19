"""Copy the campaign's trajectories to a viewing location, with a README and
ready-to-open PyMOL sessions and movies.

``python -m aamd.export [DEST] [--no-copy] [--no-movie] [--systems S ...]``

Run with the ``viz`` environment (PyMOL, ffmpeg, mdtraj; see
``aamd/environment.yml``), which needs nothing from OpenMM:

    .aamd-env/mamba/envs/viz/bin/python -m aamd.export /mnt/d/collSIm-aamd-trajectories

Layout under DEST mirrors ``aamd/work``::

    DEST/README.txt
    DEST/<system>/system.pdb, segments.pdb         built system (with water), the two segments
    DEST/<system>/<run>/solute.pdb                  topology of the DCDs (protein + ions, no water)
    DEST/<system>/<run>/solute_chains.pdb           the same atoms with chain ids A-F, ions chain I
    DEST/<system>/<run>/solute_NNN.dcd              production, 0.5 ns per file, a frame every 10 ps
    DEST/<system>/<contact run>/session.pse         PyMOL session of the run nearest contact
    DEST/<system>/<contact run>/movie.mp4, .gif     the same run as a movie

Chains: A, B = alpha1(I) and C = alpha2(I) of segment 1; D, E = alpha1(I) and
F = alpha2(I) of segment 2 (the partner, staggered). Each chain is ACE + 30
residues + NME.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

AAMD = Path(__file__).resolve().parent
WORK = AAMD / "work"
RESULTS = AAMD / "results"
DEFAULT_DEST = Path("/mnt/d/collSIm-aamd-trajectories")
CHAIN_IDS = "ABCDEF"
RES_PER_CHAIN = 32                        # ACE + 30 + NME
IONS = ("Na+", "Cl-", "MG", "CA", "SO4")
ACIDIC = ("ASP", "GLU", "ASH", "GLH", "SLL", "SLH")
BASIC = ("ARG", "LYS", "LYN", "HIS", "HIE", "HID", "HIP")
CHEM = {"native": "native (unmodified) collagen, pH 7.4",
        "succ": "fully succinylated lysines, pH 4.2 (carboxyl protonation from the recommended pKa set)",
        "acet": "fully acetylated lysines, pH 4.2"}
DIVALENT = {None: "150 mM NaCl only", "Ca": "150 mM NaCl + 50 mM CaCl2", "Mg": "150 mM NaCl + 50 mM MgCl2",
            "SO4": "150 mM NaCl + 50 mM Na2SO4"}


# --------------------------------------------------------------- metadata ---
def systems():
    return sorted(d.name for d in WORK.iterdir() if (d / "build.json").exists())


def runs(name):
    """Run directories with trajectory data: wNN windows and cR.RRR runs."""
    out = []
    for d in sorted((WORK / name).iterdir()):
        if d.is_dir() and (d.name.startswith("w") or d.name.startswith("c")) and list(d.glob("solute_*.dcd")):
            out.append(d)
    return out


def run_info(d: Path):
    """centre r0, mean and sd of the separation, production ns, finished?"""
    done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else None
    prog = json.loads((d / "progress.json").read_text()) if (d / "progress.json").exists() else {}
    r0 = (done or prog).get("r0")
    cv = [np.load(f) for f in sorted(d.glob("cv_*.npy"))]
    x = np.concatenate([c[:, 1] for c in cv]) if cv else np.array([np.nan])
    return {"run": d.name, "r0_nm": r0, "mean_nm": float(np.mean(x)), "sd_nm": float(np.std(x)),
            "ns": len(list(d.glob("solute_*.dcd"))) * 0.5, "finished": done is not None}


def describe_system(name):
    spec = json.loads((WORK / name / "build.json").read_text())["spec"]
    stagger = {234: "1 D (234 residues, the native D-period register)",
               117: "0.5 D (117 residues, a disfavoured register)"}.get(spec["stagger_res"],
                                                                      f"{spec['stagger_res']} residues")
    return (f"{name}: helix residues {spec['start_a']}-{spec['start_a'] + spec['n_res'] - 1} against "
            f"{spec['start_a'] - spec['stagger_res']}-{spec['start_a'] - spec['stagger_res'] + spec['n_res'] - 1}"
            f" (stagger {stagger}); {CHEM[spec['chemistry']]}; {DIVALENT[spec['divalent']]}")


def contact_run(name):
    """The run nearest contact: mean separation closest to the PMF minimum
    (aamd/results/comparison.json), else the run with the smallest centre."""
    rs = [run_info(d) for d in runs(name)]
    if not rs:
        return None
    d_min = None
    cmp_file = RESULTS / "comparison.json"
    if cmp_file.exists():
        for row in json.loads(cmp_file.read_text()).get("rows", []):
            if row["system"] == name:
                d_min = row["d_contact_nm"]
    if d_min is None:
        return min(rs, key=lambda r: r["r0_nm"])["run"]
    return min(rs, key=lambda r: abs(r["mean_nm"] - d_min))["run"]


# ------------------------------------------------------------------ files ---
def chained_pdb(src: Path, dst: Path):
    """solute.pdb with chain ids: residues in order, 32 per chain, then ions
    (chain I). The atom order is unchanged, so the DCDs still match."""
    out, res_seen, last = [], 0, None
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[17:20], line[22:27])
            if key != last:
                res_seen += 1
                last = key
            resn = line[17:20].strip()
            k = (res_seen - 1) // RES_PER_CHAIN
            chain = "I" if resn in IONS or k >= len(CHAIN_IDS) else CHAIN_IDS[k]
            line = line[:21] + chain + line[22:]
        out.append(line)
    dst.write_text("\n".join(out) + "\n")


def copy_tree(dest: Path, names, log=print):
    for name in names:
        sd, dd = WORK / name, dest / name
        dd.mkdir(parents=True, exist_ok=True)
        for f in ("system.pdb", "segments.pdb"):
            if (sd / f).exists():
                shutil.copy2(sd / f, dd / f)
        for r in runs(name):
            rd = dd / r.name
            rd.mkdir(exist_ok=True)
            for f in [r / "solute.pdb", *sorted(r.glob("solute_*.dcd"))]:
                t = rd / f.name
                if not t.exists() or t.stat().st_size != f.stat().st_size or t.stat().st_mtime < f.stat().st_mtime:
                    shutil.copy2(f, t)
            chained_pdb(r / "solute.pdb", rd / "solute_chains.pdb")
        log(f"copied {name}: {len(runs(name))} runs")


# --------------------------------------------------------------- sessions ---
def make_session(run_dir: Path, out_dir: Path, title: str, movie=True, n_movie=150, threads=2):
    """PyMOL session (and movie) of one run, from the copied files in out_dir."""
    from pymol import cmd

    cmd.reinitialize()
    cmd.feedback("disable", "all", "everything")
    cmd.set("max_threads", threads)
    dcds = sorted(out_dir.glob("solute_*.dcd"))
    cmd.load(str(out_dir / "solute_chains.pdb"), "pair")
    n_frames = 50 * len(dcds)
    stride = max(1, n_frames // 400)                 # at most ~400 states in the session
    for k, f in enumerate(dcds):
        # the first file replaces state 1 (the built start structure), the rest append
        cmd.load_traj(str(f), "pair", state=1 if k == 0 else 0, interval=stride)
    cmd.remove("pair and hydro and not (resn SO4)")
    cmd.intra_fit("pair and name CA and chain A+B+C")
    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.alter("pair and polymer", "ss='L'")       # a plain tube through the CA trace
    cmd.cartoon("tube", "pair")
    cmd.set("cartoon_tube_radius", 0.4)
    cmd.show("cartoon", "pair and polymer")
    # the three chains of each segment distinct; partner segment in darker shades
    for ch, col in zip("ABCDEF", ("palegreen", "lightorange", "lightpink", "forest", "orange", "hotpink")):
        cmd.color(col, f"pair and chain {ch}")
    # whole side chains from CA, so that the sticks join the backbone tube
    acid = "pair and resn ASP+ASH+GLU+GLH+SLL+SLH and not name N+C+O"
    base = "pair and resn ARG+LYS+LYN+HIS+HIE+HID+HIP and not name N+C+O"
    cmd.select("acidic", acid)
    cmd.select("basic", base)
    cmd.show("sticks", "acidic or basic")
    cmd.set("stick_radius", 0.25)
    cmd.color("red", "acidic")
    cmd.color("blue", "basic")
    cmd.select("divalent", "pair and resn CA+MG and chain I")
    cmd.select("sulfate", "pair and resn SO4")
    cmd.show("spheres", "divalent or sulfate")
    cmd.set("sphere_scale", 0.8, "divalent")
    cmd.color("yellow", "pair and resn CA and chain I")
    cmd.color("cyan", "pair and resn MG and chain I")
    cmd.set("sphere_scale", 0.45, "sulfate")
    cmd.color("purple", "sulfate and elem S")
    cmd.color("violet", "sulfate and elem O")
    cmd.select("monovalent", "pair and resn Na++Cl-")
    cmd.deselect()
    # camera along y: helix axes (z) horizontal, the separation (x) vertical
    view = list(cmd.get_view())
    view[:9] = [0, 1, 0, 0, 0, 1, 1, 0, 0]
    cmd.set_view(view)
    cmd.zoom("pair and polymer", 2, state=-1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 1)
    cmd.set_title("pair", 1, title)
    cmd.save(str(out_dir / "session.pse"))
    if not movie:
        return
    tmp = out_dir / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    n_states = cmd.count_states("pair")
    states = np.unique(np.linspace(1, n_states, min(n_movie, n_states)).astype(int))
    for k, s in enumerate(states):
        cmd.frame(int(s))
        cmd.set("state", int(s))
        cmd.png(str(tmp / f"f{k:04d}.png"), width=960, height=540, ray=1)
    ff = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "15", "-i", str(tmp / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=960:-2", str(out_dir / "movie.mp4")],
                   check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "12", "-i", str(tmp / "f%04d.png"),
                    "-vf", "scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                    str(out_dir / "movie.gif")], check=True)
    shutil.rmtree(tmp)


# ----------------------------------------------------------------- README ---
README_HEAD = """collSIm explicit-solvent campaign: trajectories
================================================

Pairs of 30-residue type I collagen triple-helical segments (human COL1A1 x2 +
COL1A2) side by side in water, held at a fixed inter-axial separation d by an
umbrella restraint (OpenMM 8.6, ff14SB + TIP3P, 298 K, 1 bar). Method, force
field, restraints and results: docs/explicit_solvent_pmf.md in the collSIm
repository (branch fm/collsim-aamd-pmf), code in aamd/.

Layout
------
<system>/system.pdb        the built system, with water and ions (start structure)
<system>/segments.pdb      the two segments before solvation
<system>/<run>/solute.pdb          topology for the DCDs (protein + all ions, no water)
<system>/<run>/solute_chains.pdb   the same atoms with chain ids (use this one)
<system>/<run>/solute_NNN.dcd      production, 0.5 ns per file, one frame per 10 ps
<system>/<run>/session.pse         PyMOL session (only in the run nearest contact)
<system>/<run>/movie.mp4, movie.gif  the same run as a movie

Runs: wNN = umbrella window NN with centre d = 1.4 + 0.1*NN nm; cR.RRR = an
extra run with centre R.RRR nm (c1.300: the inner window at 1.3 nm, the
contact; ion-only systems have just one such contact run). Production files
are consecutive in time; water is not saved. Coordinates are not wrapped into
the box, so the molecules stay whole.

Chains: A, B = alpha1(I), C = alpha2(I) of segment 1 (helix residues 624-653);
D, E = alpha1(I), F = alpha2(I) of segment 2 (the staggered partner). Each
chain is ACE + 30 residues + NME. Ions are chain I (Na+, Cl-, CA = Ca2+,
MG = Mg2+, SO4 = sulfate). Residue names: HYP hydroxyproline, SLL succinyl-
lysine (carboxylate), SLH succinyl-lysine (carboxylic acid), ASH/GLH
protonated Asp/Glu, ALY acetyl-lysine.

Colours in the sessions and movies: chains A/B/C pale green / light orange /
light pink, D/E/F dark green / orange / hot pink; Asp/Glu/succinyl
side chains red sticks; Arg/Lys/His side chains blue sticks; Ca2+ yellow
spheres, Mg2+ cyan spheres, sulfate purple/violet spheres; Na+ and Cl- hidden
(selection "monovalent").

Opening
-------
Session:   double-click session.pse (PyMOL), or in PyMOL: File > Open. Play
           with the movie buttons at the bottom right.
Movie:     movie.mp4 in any video player; movie.gif in a browser.

Trajectory, PyMOL:
    load solute_chains.pdb, pair
    load_traj solute_000.dcd, pair, 0
    load_traj solute_001.dcd, pair, 0     (state 0 appends; and so on, or in a loop:)
    python
    import glob
    for f in sorted(glob.glob("solute_*.dcd")): cmd.load_traj(f, "pair", state=0, interval=5)
    python end
    intra_fit pair and name CA and chain A+B+C

Trajectory, ChimeraX:
    open solute_chains.pdb
    open solute_*.dcd structureModel #1
    (then the Movie / coordset slider; "coordset #1" plays it)

Trajectory, VMD:
    vmd solute_chains.pdb solute_000.dcd solute_001.dcd ...
    or: mol new solute_chains.pdb; foreach f [lsort [glob solute_*.dcd]] { mol addfile $f waitfor all }

Python (mdtraj):
    import mdtraj, glob
    t = mdtraj.load(sorted(glob.glob("solute_*.dcd")), top="solute_chains.pdb")

Systems and runs
----------------
"""


def write_readme(dest: Path, names, sessions):
    lines = [README_HEAD]
    for name in names:
        lines.append(describe_system(name))
        for r in runs(name):
            i = run_info(r)
            tag = "   <- session.pse, movie" if sessions.get(name) == r.name else ""
            lines.append(f"    {i['run']:<8s} centre {i['r0_nm']:.2f} nm, mean d {i['mean_nm']:.3f} +- "
                         f"{i['sd_nm']:.3f} nm, {i['ns']:.1f} ns{'' if i['finished'] else ' (unfinished)'}{tag}")
        lines.append("")
    (dest / "README.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default=str(DEFAULT_DEST))
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--no-session", action="store_true")
    ap.add_argument("--systems", nargs="*")
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = a.systems or systems()
    names = [n for n in names if runs(n)]
    if not a.no_copy:
        copy_tree(dest, names)
    sessions = {n: contact_run(n) for n in names}
    if not a.no_session:
        for n in names:
            r = sessions[n]
            print(f"session {n} {r}", flush=True)
            make_session(WORK / n / r, dest / n / r, f"{n} {r}", movie=not a.no_movie)
    write_readme(dest, systems(), sessions)
    print(f"README: {dest / 'README.txt'}")


if __name__ == "__main__":
    main()
