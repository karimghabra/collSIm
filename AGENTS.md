# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Python work (`modcoll/`, `aamd/`) needs numba; `tests/modcoll` runs in `.venv` (see `requirements-modcoll.txt`), `tests/aamd` in the conda-forge environment of `aamd/environment.lock.txt` (OpenMM CUDA, AmberTools, psi4) with its `bin/` on PATH, or `test_md.py` skips for lack of tleap.
- `aamd/` all-atom campaign: control with `aamd/aamd status | run | stop`; jobs resume from their last 0.5 ns chunk; checks, thresholds and method in `docs/explicit_solvent_pmf.md`. Trajectories live in git-ignored `aamd/work/`.
- In ff14SB the carbonyl O and carboxylate O2 share one LJ type, so a per-type 12-6-4 C4 table cannot target carboxylates; `aamd/md.py` applies it per atom.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
