# Repository Guidelines

## Project Structure & Modules
- Code: `rtrg/` with domains `core/`, `field_theory/`, `renormalization/`, `israel_stewart/`, plus `visualization/`, `numerics/`, `analysis/`, `symbolic/`, `production/`.
- Tests: `tests/unit/`, `tests/integration/`, `tests/benchmarks/` (markers: unit, integration, physics, numerical, slow, benchmark). Shared fixtures live in `tests/conftest.py`.
- Docs & Plans: `docs/` (api/theory/tutorials), design notes in `plan/`.
- Config: `pyproject.toml`, `pytest.ini`, `.pre-commit-config.yaml`, `uv.lock`.

## Build, Test, and Development Commands
- Environment: `uv sync --all-extras --dev` (installs dev/test tools).
- Lint/Format: `uv run ruff check --fix` and `uv run ruff format`.
- Types: `uv run mypy rtrg/`.
- Tests (all): `uv run pytest -v`.
- Tests (by suite): `uv run pytest tests/unit/ -m "not slow"`; integration: `uv run pytest tests/integration/`.
- Coverage (CI mirrors): `uv run pytest --cov=rtrg --cov-report=term-missing --cov-fail-under=70`.
- CLI sanity: `uv run relativistic-turbulence-rg`.

## Coding Style & Naming
- Python 3.12+, 4‑space indentation, max line length 100.
- Use type hints; MyPy is strict (see `[tool.mypy]`).
- Ruff is the source of truth for linting/formatting. Run hooks locally: `pre-commit run -a`.
- Naming: modules `lower_snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Tests: files `test_*.py`, classes `Test*`, functions `test_*`.

## Testing Guidelines
- Write unit tests near the module’s domain (e.g., `tests/unit/test_tensors.py`).
- Mark appropriately: `@pytest.mark.unit`, `integration`, `physics`, `numerical`, `slow`.
- Aim for ≥70% coverage (enforced in CI). Prefer fast, deterministic tests; gate slow paths with `-m "slow"`.
- Run subsets: `uv run pytest -m "unit and not slow"`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., "Fix tensor symmetrization edge case"). Reference module or test when helpful.
- PRs: include a clear description, linked issues, test evidence (new/updated tests), and any numerical/physics validation notes. CI must pass (ruff, mypy, pytest, coverage).
- Avoid large files in PRs; prefer scripts/figures that can be generated.

## Security & CI Tips
- Run `bandit -c pyproject.toml -r rtrg/` for a quick scan (CI runs this).
- Supported Python versions in CI: 3.12, 3.13. Keep changes compatible.

## Physics Validation
- Constraints: enforce `u_μ u^μ = -c^2`; use spatial projector `Δ^{μν}` and keep `π^{μν}` symmetric, traceless, spatial.
- Linear dispersion: verify small-k dispersion and attenuation `Γ = (4η/3 + ζ)/ρ`.
- Bjorken flow: compare against boost-invariant analytical evolution for sanity checks.
- Relaxation dynamics: check time scales `τ_π, τ_Π, τ_q` in equilibration tests; ensure hyperbolicity and CFL with relaxation.
- Stochastic consistency: when present, validate FDT-consistent noise correlators with tensor projectors.
- Run suite: `uv run pytest -m "physics"` (or `-m "physics and not slow"`).

### MSRJD Notes
- Response fields: include tilde fields (e.g., `~ρ, ~u_μ, ~π_{μν}`) and verify conservation, relaxation, and noise terms appear in the action with correct signs.
- Projectors: use `Δ^{μν}` and transverse-traceless `P_{μναβ}` to enforce spatial, traceless structure in propagators and noise correlators.
- Noise checks: confirm Keldysh sector matches FDT (e.g., `D^π = 2 k_B T η P_{μναβ}`, `D^q_{μν} = 2 k_B T^2 κ Δ_{μν}`) and maintains Lorentz covariance.
- Reference: see `plan/MSRJD_Formalism.md` for derivations, propagators, and RG setup.
