"""Throughput of one umbrella window on this machine: ``python -m aamd.bench SYSTEM``."""
import sys
import time

import numpy as np
from openmm import unit

from . import build, md


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0]
    steps = int(argv[1]) if len(argv) > 1 else 25000
    wd = build.WORK / name
    sim, groups, _ = md.make_simulation(wd / "system.prmtop", wd / "system.inpcrd",
                                        r0=build.SYSTEMS[name].d0_nm)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(md.TEMPERATURE * unit.kelvin)
    sim.step(2500)
    t0 = time.time()
    sim.step(steps)
    st = sim.context.getState(getPositions=True)
    elapsed = time.time() - t0
    dt_fs = sim.integrator.getStepSize().value_in_unit(unit.femtosecond)
    ns_day = steps * dt_fs * 1e-6 / (elapsed / 86400.0)
    off = md.centroid_offsets(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer), groups)
    print(f"{name}: {sim.system.getNumParticles()} atoms, {steps} steps of {dt_fs} fs in "
          f"{elapsed:.1f} s -> {ns_day:.0f} ns/day; centroid offset {np.round(off, 3)} nm")


if __name__ == "__main__":
    main()
