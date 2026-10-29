# Agent notes

- Environment: editable install in the `ldpred3-accelerate` conda env (`~/anaconda3/envs/ldpred3-accelerate/bin/python`, 3.14), with Numba and SciPy.
- Run Python with `-P` when the working directory is `REPOS/`. Otherwise the `allelign/` repo directory shadows the installed package as a namespace package, and `import allelign` finds no `__version__`.
- Tests: `pytest -q`, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for deterministic numerics. Markers:
  - `parity`: compares against ldpred3; skipped without it;
  - `numba`: needs the JIT.
- `NUMBA_DISABLE_JIT=1 pytest -m "not parity"` checks the pure-Python fallback.
- The parity tests are the contract for moving ldpred3 and its siblings onto allelign. A change that breaks them needs a deliberate decision, recorded in docs/guide.md section 5.
- Lint is `ruff check .`, pyflakes rules only. Do not add style rules or reformat long lines.
- Status codes, warning-marker strings and the AF safety-valve wording are read by other repos (SMARTpred). Change them only together with their consumers.
