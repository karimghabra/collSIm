"""Umbrella sampling along the inter-axial separation, with live checks.

``python -m aamd.umbrella pull SYSTEM``           equilibrate at the built separation, pull
                                                  inwards, save one start state per window
``python -m aamd.umbrella window SYSTEM I``       run umbrella window I
``python -m aamd.umbrella contact SYSTEM R0 NS``  one long window at separation R0
``python -m aamd.umbrella worker K N JOBFILE``    worker K of N: take ready jobs from JOBFILE until none is left

Protocol (``Protocol``): windows at r0 = 1.40 ... 2.80 nm every 0.10 nm,
k_umb = 1000 kJ/mol/nm^2 (thermal width 0.05 nm, half the spacing), 0.5 ns of
equilibration discarded, 4 ns of production in 0.5 ns chunks, CV and energies
written every 1 ps, solute and ion coordinates every 10 ps. NPT, 298.15 K,
1 bar, 4 fs with hydrogen mass repartitioning (3 amu), Langevin middle
integrator at 1/ps.

The pull: 0.2 ns NVT + 0.3 ns NPT at r0 = 3.0 nm, then r0 is moved linearly
to 1.15 nm over 2 ns; the state is saved as r0 passes each window centre.

A job (window or contact run) writes into its own directory
``aamd/work/SYSTEM/wNN/`` (or ``cR.RRR/``):

  cv_XXX.npy      per chunk, per ps: time, Delta x, Delta y, Delta z (nm),
                  potential and kinetic energy (kJ/mol), temperature (K), volume (nm^3)
  solute_XXX.dcd  per chunk, solute + ions every 10 ps, not wrapped
  state.xml       restart point, rewritten atomically after each chunk
  progress.json   chunks done
  status.json     updated after equilibration and every chunk: ns done, ns/day,
                  ETA, last checkpoint, worst check verdict
  checks.json     every check of every chunk
  failed.json     present if a hard check failed; the job stops
  done.json       present when finished

Every warning or failure is also one line in ``aamd/work/WARNINGS``.
Restart: rerun the same job (or ``python -m aamd run``); finished chunks are
kept and the job resumes from state.xml.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

from . import build, checks, md

WARNINGS = build.WORK / "WARNINGS"


@dataclasses.dataclass(frozen=True)
class Protocol:
    r_min: float = 1.40
    r_max: float = 2.80
    dr: float = 0.10
    k_umb: float = 1000.0
    eq_ns: float = 0.5
    prod_ns: float = 4.0
    chunk_ns: float = 0.5
    cv_every_ps: float = 1.0
    dcd_every_ps: float = 10.0
    dt_fs: float = 4.0
    pull_start: float = 3.0
    pull_end: float = 1.15
    pull_ns: float = 2.0

    @property
    def centres(self):
        n = int(round((self.r_max - self.r_min) / self.dr)) + 1
        return np.round(self.r_min + self.dr * np.arange(n), 4)

    def steps(self, ns):
        return int(round(ns * 1e6 / self.dt_fs))


PROTOCOL = Protocol()


class JobFailed(RuntimeError):
    pass


def workdir(name):
    return build.WORK / name


def _sim(name, r0, protocol=PROTOCOL, restraint=None):
    wd = workdir(name)
    spec = md.RestraintSpec(k_umb=protocol.k_umb) if restraint is None else restraint
    return md.make_simulation(wd / "system.prmtop", wd / "system.inpcrd", r0=r0,
                              restraint=spec, dt_fs=protocol.dt_fs)


def _positions(sim):
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def solute_and_ions(topology, groups):
    return sorted(set(groups.solute.tolist()) | {a.index for a in topology.atoms()
                                                  if a.residue.name not in ("WAT", "HOH")})


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _save_state(sim, path: Path):
    tmp = path.with_suffix(".tmp.xml")
    sim.saveState(str(tmp))
    os.replace(tmp, path)


def degrees_of_freedom(system) -> int:
    n = sum(1 for i in range(system.getNumParticles()) if system.getParticleMass(i).value_in_unit(unit.dalton) > 0)
    dof = 3 * n - system.getNumConstraints()
    if any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces()):
        dof -= 3
    return dof


# --------------------------------------------------------------------- pull ---
def pull(name, protocol=PROTOCOL, centres=None):
    wd = workdir(name)
    out = wd / "windows"
    out.mkdir(exist_ok=True)
    status = wd / "pull_status.json"
    t_start = time.time()
    sim, groups, prmtop = _sim(name, protocol.pull_start, protocol)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(md.TEMPERATURE * unit.kelvin, 1)
    baro = next(f for f in sim.system.getForces() if isinstance(f, openmm.MonteCarloBarostat))
    baro.setFrequency(0)
    sim.context.reinitialize(preserveState=True)
    sim.step(protocol.steps(0.2))
    baro.setFrequency(25)
    sim.context.reinitialize(preserveState=True)
    sim.step(protocol.steps(0.3))
    log = []
    centres = list((protocol.centres if centres is None else np.asarray(centres))[::-1])
    all_c = protocol.centres if centres is None else np.asarray(centres)
    n_pull = protocol.steps(protocol.pull_ns)
    per_ps = protocol.steps(0.001)
    n_upd = n_pull // per_ps
    for k in range(n_upd + 1):
        r0 = protocol.pull_start + (protocol.pull_end - protocol.pull_start) * k / n_upd
        sim.context.setParameter("r0", r0)
        sim.step(per_ps)
        if k % 10 == 0:
            off = md.centroid_offsets(_positions(sim), groups)
            if not np.all(np.isfinite(off)):
                raise JobFailed(f"{name}: pull produced non-finite coordinates at r0={r0:.3f}")
            log.append([k, r0, *off])
        if k % 200 == 0:
            el = time.time() - t_start
            _atomic_write(status, json.dumps({"job": f"pull {name}", "fraction": k / n_upd,
                                              "r0": r0, "wall_s": el, "updated": time.time()}))
        while centres and r0 <= centres[0]:
            c = centres.pop(0)
            i = int(np.argmin(np.abs(all_c - c)))
            _save_state(sim, out / f"start_{i:02d}.xml")
    np.save(wd / "pull_log.npy", np.array(log))
    (wd / "pull.json").write_text(json.dumps({"protocol": dataclasses.asdict(protocol),
                                              "wall_s": time.time() - t_start}, indent=1))
    _atomic_write(status, json.dumps({"job": f"pull {name}", "fraction": 1.0, "done": True,
                                      "updated": time.time()}))


# ------------------------------------------------------------- one window ---
def _lock(jobdir: Path):
    """Exclusive lock file holding the pid; a stale lock (dead pid) is taken over."""
    lock = jobdir / "lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip() or 0)
            os.kill(pid, 0)
            return False                      # alive: someone else runs it
        except (ProcessLookupError, ValueError):
            lock.write_text(str(os.getpid()))
            return True
        except PermissionError:
            return False


def _unlock(jobdir: Path):
    try:
        (jobdir / "lock").unlink()
    except FileNotFoundError:
        pass


def run_job(name, jobdir: Path, r0: float, prod_ns: float, start_state: Path,
            protocol=PROTOCOL, restraint=None, label=None):
    """Equilibrate, then produce ``prod_ns`` in chunks with checks after each."""
    jobdir.mkdir(parents=True, exist_ok=True)
    label = label or f"{name}/{jobdir.name}"
    if (jobdir / "done.json").exists() or (jobdir / "failed.json").exists():
        return
    if not _lock(jobdir):
        return
    try:
        _run_job(name, jobdir, r0, prod_ns, start_state, protocol, restraint, label)
    finally:
        _unlock(jobdir)


def _run_job(name, jobdir, r0, prod_ns, start_state, protocol, restraint, label):
    wd = workdir(name)
    sim, groups, prmtop = _sim(name, r0, protocol, restraint)
    dof = degrees_of_freedom(sim.system)
    z0 = sim.context.getParameter("z0")
    subset = solute_and_ions(prmtop.topology, groups)
    sub_top = _subset_topology(prmtop.topology, subset)
    ref_file = wd / "integrity_ref.json"
    if not ref_file.exists():
        x0 = app.AmberInpcrdFile(str(wd / "system.inpcrd")).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        ref = checks.build_integrity_ref(sub_top, x0[subset])
        _atomic_write(ref_file, json.dumps(ref.to_json()))
    ref = checks.IntegrityRef.from_json(json.loads(ref_file.read_text()))
    if not (jobdir / "solute.pdb").exists():
        x0 = app.AmberInpcrdFile(str(wd / "system.inpcrd")).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        with open(jobdir / "solute.pdb", "w") as fh:
            app.PDBFile.writeFile(sub_top, x0[subset] * 10.0 * unit.angstrom, fh)
    # static check: ions present as built
    found = {}
    for r in prmtop.topology.residues():
        if r.name in ("Na+", "Cl-", "MG", "CA", "SO4"):
            key = r.name
            found[key] = found.get(key, 0) + 1
    expected = json.loads((wd / "build.json").read_text())["ions"]
    expected = {k: v for k, v in expected.items() if v}
    c_ion = checks.check_ion_count(found, expected)
    ion_idx = {"Mg": [a.index for a in prmtop.topology.atoms() if a.residue.name == "MG"],
               "Ca": [a.index for a in prmtop.topology.atoms() if a.residue.name == "CA" and a.name == "CA"
                      and a.residue.name == "CA"]}
    oxy_idx = np.array([a.index for a in prmtop.topology.atoms() if a.element is not None
                        and a.element.symbol == "O"])

    state, progress = jobdir / "state.xml", jobdir / "progress.json"
    if state.exists() and progress.exists():
        sim.loadState(str(state))
        prog = json.loads(progress.read_text())
    else:
        sim.loadState(str(start_state))
        sim.context.setParameter("r0", r0)
        t0 = time.time()
        try:
            sim.step(protocol.steps(protocol.eq_ns))
        except openmm.OpenMMException as ex:
            c = checks.Check("finite", 1.0, "-", "any", "fail", f"equilibration stopped: {str(ex)[:160]}")
            checks.record([c], f"{label} equilibration", WARNINGS)
            _atomic_write(jobdir / "checks.json", json.dumps([{"chunk": -1, "checks": [dataclasses.asdict(c)]}]))
            _atomic_write(jobdir / "failed.json", json.dumps({"chunk": -1, "failed": ["finite"]}))
            raise JobFailed(f"{label}: equilibration stopped: {ex}")
        prog = {"chunk": 0, "r0": r0, "k_umb": protocol.k_umb, "eq_wall_s": time.time() - t0,
                "prev_pe": None, "prev_vol": None}
        _save_state(sim, state)
        _atomic_write(progress, json.dumps(prog))
    sim.context.setParameter("r0", r0)
    n_chunks = int(round(prod_ns / protocol.chunk_ns))
    per_cv = protocol.steps(protocol.cv_every_ps * 1e-3)
    per_dcd = protocol.steps(protocol.dcd_every_ps * 1e-3)
    all_checks = json.loads((jobdir / "checks.json").read_text()) if (jobdir / "checks.json").exists() else []
    walls = prog.get("walls", [])
    while prog["chunk"] < n_chunks:
        c = prog["chunk"]
        dcd_path = jobdir / f"solute_{c:03d}.dcd"
        dcd = app.DCDReporter(str(dcd_path), per_dcd, atomSubset=subset, enforcePeriodicBox=False)
        sim.reporters = [dcd]
        n = protocol.steps(protocol.chunk_ns) // per_cv
        rec = np.zeros((n, 8))
        t_chunk = time.time()
        t_start = c * protocol.chunk_ns * 1000.0
        crashed = None
        for k in range(n):
            try:
                sim.step(per_cv)
            except openmm.OpenMMException as ex:     # e.g. "Particle coordinate is NaN"
                crashed = str(ex)
                rec = rec[:k]
                break
            st = sim.context.getState(getPositions=True, getEnergy=True)
            pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            rec[k, 0] = t_start + (k + 1) * protocol.cv_every_ps
            rec[k, 1:4] = md.centroid_offsets(pos, groups)
            rec[k, 4] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            rec[k, 5] = ke
            rec[k, 6] = 2.0 * ke / (dof * 0.0083144626)
            rec[k, 7] = st.getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
            if not np.all(np.isfinite(rec[k])):
                rec = rec[:k + 1]
                break
        sim.reporters = []
        del dcd
        wall = time.time() - t_chunk
        np.save(jobdir / f"cv_{c:03d}.npy", rec)
        # ---- checks on this chunk
        where = f"{label} chunk {c}"
        if crashed is not None:
            cs = [checks.Check("finite", 1.0, "-", "any", "fail", f"integrator stopped: {crashed[:160]}")]
        else:
            cs = [checks.check_finite(rec, pos)]
        if cs[0].verdict == "ok":
            cs += checks.check_temperature(rec[:, 6])
            cs.append(checks.check_drift("energy_drift", float(rec[:, 4].mean()), prog.get("prev_pe"), 0.003, 0.02))
            cs.append(checks.check_drift("volume_drift", float(rec[:, 7].mean()), prog.get("prev_vol"), 0.01, 0.03))
            cs += checks.check_umbrella(rec[:, 1:4], r0, z0)
            try:
                import mdtraj
                frames = mdtraj.load_dcd(str(dcd_path), top=str(jobdir / "solute.pdb")).xyz
                cs += checks.check_structure(frames, ref)
            except Exception as ex:                       # an unreadable chunk is itself a failure
                cs.append(checks.Check("dcd_readable", 0.0, "-", "unreadable", "fail", repr(ex)[:200]))
            cs.append(c_ion)
            box = np.diag(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            cs += checks.check_ion_coordination(
                checks.coordination(pos, ion_idx["Mg"], oxy_idx, box, 0.26),
                checks.coordination(pos, ion_idx["Ca"], oxy_idx, box, 0.30))
        worst = checks.record(cs, where, WARNINGS)
        all_checks.append({"chunk": c, "checks": [dataclasses.asdict(x) for x in cs]})
        _atomic_write(jobdir / "checks.json", json.dumps(all_checks))
        if worst == "fail":
            failed = [x.name for x in cs if x.verdict == "fail"]
            _atomic_write(jobdir / "failed.json", json.dumps({"chunk": c, "failed": failed,
                                                               "time": time.time()}))
            _status(jobdir, label, r0, prog["chunk"] * protocol.chunk_ns, prod_ns, walls, protocol,
                    "failed", failed)
            raise JobFailed(f"{where}: hard check failure {failed}")
        walls.append(wall)
        prog.update(chunk=c + 1, prev_pe=float(rec[:, 4].mean()), prev_vol=float(rec[:, 7].mean()),
                    walls=walls)
        _save_state(sim, state)
        _atomic_write(progress, json.dumps(prog))
        _status(jobdir, label, r0, prog["chunk"] * protocol.chunk_ns, prod_ns, walls, protocol,
                "running" if prog["chunk"] < n_chunks else "done", worst=worst)
    _atomic_write(jobdir / "done.json", json.dumps({"r0": r0, "k_umb": protocol.k_umb, "prod_ns": prod_ns}))


def _status(jobdir, label, r0, ns_done, ns_total, walls, protocol, state, failed=None, worst=None):
    ns_day = (protocol.chunk_ns * len(walls)) / (sum(walls) / 86400.0) if walls else None
    eta_h = ((ns_total - ns_done) / ns_day * 24.0) if ns_day else None
    _atomic_write(jobdir / "status.json", json.dumps({
        "job": label, "state": state, "r0": r0, "ns_done": ns_done, "ns_total": ns_total,
        "ns_per_day": ns_day, "eta_h": eta_h, "last_checkpoint": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated": time.time(), "pid": os.getpid(), "worst_check": worst, "failed": failed}))


def window(name, i, protocol=PROTOCOL):
    wd = workdir(name)
    run_job(name, wd / f"w{i:02d}", float(protocol.centres[i]), protocol.prod_ns,
            wd / "windows" / f"start_{i:02d}.xml", protocol)


def contact(name, r0, prod_ns, protocol=PROTOCOL):
    """One long window at separation r0, started from the nearest pull state."""
    wd = workdir(name)
    i = int(np.argmin(np.abs(protocol.centres - r0)))
    run_job(name, wd / f"c{r0:.3f}", float(r0), float(prod_ns), wd / "windows" / f"start_{i:02d}.xml",
            protocol)


def _subset_topology(top, subset):
    """Topology of the atoms in ``subset`` (OpenMM has no built-in)."""
    keep = set(subset)
    new = app.Topology()
    amap = {}
    for ch in top.chains():
        nc = None
        for r in ch.residues():
            atoms = [a for a in r.atoms() if a.index in keep]
            if not atoms:
                continue
            if nc is None:
                nc = new.addChain(ch.id)
            nr = new.addResidue(r.name, nc, r.id)
            for a in atoms:
                amap[a.index] = new.addAtom(a.name, a.element, nr)
    for b in top.bonds():
        if b[0].index in amap and b[1].index in amap:
            new.addBond(amap[b[0].index], amap[b[1].index])
    return new


# ------------------------------------------------------------------ workers ---
def read_jobs(jobfile):
    return [line.split() for line in open(jobfile) if line.strip() and not line.startswith("#")]


def job_dir(job) -> Path:
    kind, name, *rest = job
    if kind == "window":
        return workdir(name) / f"w{int(rest[0]):02d}"
    if kind == "contact":
        return workdir(name) / f"c{float(rest[0]):.3f}"
    return workdir(name)


def job_finished(job) -> bool:
    kind, name, *rest = job
    d = job_dir(job)
    if kind == "build":
        return (d / "system.prmtop").exists() or (d / "failed_build.json").exists()
    if kind == "pull":
        return (d / "pull.json").exists() or (d / "failed_pull.json").exists()
    return (d / "done.json").exists() or (d / "failed.json").exists()


def job_ready(job) -> bool:
    """Prerequisites met: windows need their pull, pulls their build, builds
    of modified systems their residue libraries."""
    kind, name, *rest = job
    if kind == "build":
        spec = build.SYSTEMS[name]
        need = (["SLL", "SLH"] if spec.chemistry == "succ" else []) + (["SO4"] if spec.divalent == "SO4" else [])
        return all((build.PARAMS / f"{x}.lib").exists() for x in need)
    if kind == "pull":
        return (workdir(name) / "system.prmtop").exists()
    return (workdir(name) / "pull.json").exists()


def run_one(job):
    kind, name, *rest = job
    if kind == "build":
        d = workdir(name)
        d.mkdir(parents=True, exist_ok=True)
        if _lock(d):
            try:
                build.build(build.SYSTEMS[name])
            finally:
                _unlock(d)
            return True
        return False
    if kind == "pull":
        d = workdir(name)
        if _lock(d):
            try:
                pull(name)
            finally:
                _unlock(d)
            return True
        return False
    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "lock").exists():
        try:
            os.kill(int((d / "lock").read_text().strip() or 0), 0)
            return False                         # another worker has it
        except (ProcessLookupError, ValueError):
            pass
    if kind == "window":
        window(name, int(rest[0]))
    elif kind == "contact":
        contact(name, float(rest[0]), float(rest[1]))
    return True


def worker(k, n, jobfile):
    """Take the first job of JOBFILE that is unfinished, ready and not locked
    by a live worker; run it; re-read the file; repeat. Exits when every job
    is finished. A failed job is logged (WARNINGS) and not retried. Several
    workers share the list through the lock files; ``k`` and ``n`` only name
    the worker."""
    while True:
        jobs = read_jobs(jobfile)
        pending = [j for j in jobs if not job_finished(j)]
        if not pending:
            print(f"[worker {k}] all jobs finished", flush=True)
            return
        ran = False
        for job in pending:
            if not job_ready(job):
                continue
            try:
                if run_one(job):
                    ran = True
                    print(f"[worker {k}] done {' '.join(job)}", flush=True)
                    break
            except JobFailed as ex:
                ran = True
                if job[0] == "pull":
                    (job_dir(job) / "failed_pull.json").write_text(json.dumps({"error": str(ex)}))
                print(f"[worker {k}] FAILED {' '.join(job)}: {ex}", flush=True)
                break
            except Exception as ex:
                ran = True
                with open(WARNINGS, "a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(job)} job_error FAIL {ex!r}\n")
                fname = f"failed_{job[0]}.json" if job[0] in ("build", "pull") else "failed.json"
                (job_dir(job) / fname).write_text(json.dumps({"error": repr(ex)}))
                print(f"[worker {k}] ERROR {' '.join(job)}: {ex!r}", flush=True)
                break
        if not ran:
            time.sleep(60)


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a[0] == "pull":
        pull(a[1])
    elif a[0] == "window":
        window(a[1], int(a[2]))
    elif a[0] == "contact":
        contact(a[1], float(a[2]), float(a[3]))
    elif a[0] == "worker":
        worker(int(a[1]), int(a[2]), a[3])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
