# spikestats

Pure-Python spike-train statistics. Zero runtime dependencies. Completes the toolkit with
spikegen (generate) and spikedist (compare).

## Commands

- Create env and install: `uv venv && uv pip install -e ".[dev]"`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .` (format with `uv run ruff format .`)
- Types: `uv run mypy src`
- Build: `uv build` (then `uv run --with twine twine check dist/*` before publishing)
- Example: `python examples/summary.py`

## Architecture

`src/spikestats/`:
- `_validate.py` shared validation (positivity, non-negativity, finite times, minimum spikes)
- `metrics.py` the statistics (firing_rate, inter_spike_intervals, cv_isi, cv2, lv, lvr,
  spike_counts, fano_factor)
- `__init__.py` public surface

See `docs/architecture.md` for the formulas and references.

## Conventions

- A spike train is a plain `list[float]` of times; functions sort the input internally.
- Metrics return plain `float`; intervals and counts return lists.
- Parameters after `*` are keyword-only; no default values.
- No runtime dependencies (standard library `statistics` and `math` only). Note the module
  is `metrics.py`, not `statistics.py`, to avoid shadowing the standard library module.

## Testing rules

- Exact value for every metric against a hand-computed example.
- Known limits: a regular train gives 0, a Poisson process gives 1 (CV, Lv, Fano).
- The identity LvR(refractory=0) == Lv is asserted.
- Hypothesis property tests for bounds and invariants; bug fixes start with a failing test.

## Release

- Semantic versioning; update `CHANGELOG.md` and `__version__`.
- Gates: `uv run pytest && uv run ruff check . && uv run mypy src && uv build && uv run twine check dist/*`.
- Publish to PyPI, tag `vX.Y.Z`, GitHub release.

## Style

- No em dash characters in docs, comments, or commit messages.
- Comments explain non-obvious reasoning only.
