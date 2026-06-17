# Contributing to spikestats

Thanks for your interest. This project values correctness, clear references, and zero
runtime dependencies.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy src
```

A standard virtual environment with `pip install -e ".[dev]"` works the same way.

## Guidelines

- No runtime dependencies. The standard library `statistics` and `math` are enough.
- Every metric needs an exact-value test against a hand-computed example and, where it
  applies, a known limit (a regular train is 0, a Poisson process is 1).
- New metrics must cite the definition they implement in the docstring and in
  docs/architecture.md, so the formula can be checked against a source.
- Parameters after `*` are keyword-only with no default values.
- A bug fix starts with a failing test.
- Run `uv run ruff format .` before committing.
- Commit messages follow `type(scope): description`.

## Reporting issues

Open an issue with the spike times, the call and its parameters, and the value you expected
versus the value you observed.
