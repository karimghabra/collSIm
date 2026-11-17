"""Show that the live checks fire: three short runs of one real window.

``python -m aamd.selftest SYSTEM [WINDOW]``

* ``healthy``          the production protocol, shortened (0.02 ns equilibration, two
                       0.05 ns chunks): every check should pass;
* ``bad_timestep``     the same with a 12 fs time step (three times the production
                       step): the run heats up or blows up, and the temperature /
                       finiteness / bond checks must fail it;
* ``no_restraint``     umbrella, lateral and alignment restraints switched off
                       (k = 0) and the window centre moved 1 nm away from where the
                       start state sits: the umbrella-centre check must fail it.

Output goes to aamd/work/selftest/ (its own WARNINGS file); the summary is
printed and written to aamd/work/selftest/summary.json.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys

from . import build, md, umbrella


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    name = a[0]
    i = int(a[1]) if len(a) > 1 else 6
    root = build.WORK / "selftest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    umbrella.WARNINGS = root / "WARNINGS"
    short = dataclasses.replace(umbrella.PROTOCOL, eq_ns=0.02, chunk_ns=0.05, prod_ns=0.1)
    start = umbrella.workdir(name) / "windows" / f"start_{i:02d}.xml"
    r0 = float(umbrella.PROTOCOL.centres[i])
    cases = {
        "healthy": (short, None, r0),
        "bad_timestep": (dataclasses.replace(short, dt_fs=12.0), None, r0),
        "no_restraint": (short, md.RestraintSpec(k_umb=0.0, k_perp=0.0, k_align=0.0), r0 + 1.0),
    }
    summary = {}
    for case, (proto, restraint, rc) in cases.items():
        jd = root / case
        try:
            umbrella.run_job(name, jd, rc, proto.prod_ns, start, proto, restraint, label=f"selftest/{case}")
            outcome = "completed"
        except umbrella.JobFailed as ex:
            outcome = f"stopped: {ex}"
        except Exception as ex:          # e.g. OpenMM refusing NaN coordinates
            outcome = f"crashed: {ex!r}"[:300]
        ch = json.loads((jd / "checks.json").read_text()) if (jd / "checks.json").exists() else []
        fired = sorted({c["name"] + ":" + c["verdict"] for chunk in ch for c in chunk["checks"]
                        if c["verdict"] != "ok"})
        summary[case] = {"outcome": outcome, "non_ok_checks": fired}
        print(f"{case:<13s} {outcome[:90]}\n              non-ok: {fired}")
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"WARNINGS lines: {sum(1 for _ in open(umbrella.WARNINGS))}" if umbrella.WARNINGS.exists()
          else "WARNINGS: none")


if __name__ == "__main__":
    main()
