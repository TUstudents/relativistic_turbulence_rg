# Relativistic Turbulence Renormalization Group (RTRG) Project Context

## Project Overview

This is a Python library for applying Renormalization Group (RG) methods to analyze relativistic turbulence using the Israel-Stewart theory of relativistic viscous hydrodynamics. The project implements a complete field-theoretic framework for studying universal scaling properties in relativistic turbulent flows.

### Core Purpose
- Provide a professional implementation of RG techniques for relativistic hydrodynamics
- Enable systematic derivation of universal scaling properties in relativistic turbulence
- Implement the mathematical framework for applying RG to Israel-Stewart equations

### Key Technologies
- **Language:** Python 3.12+
- **Dependencies:**
  - Scientific computing: `numpy`, `scipy`, `sympy`
  - Performance: `numba`
  - Visualization: `matplotlib`
  - Testing: `pytest`
  - Utilities: `tqdm`
- **Build System:** `hatchling`

### Architecture
The library is organized into several core modules:

1. **`rtrg.core`** - Fundamental mathematical infrastructure
   - Tensor algebra system (`tensors`)
   - Field definitions and management (`fields`)
   - Physical parameters and constants (`parameters`, `constants`)

2. **`rtrg.israel_stewart`** - Israel-Stewart hydrodynamics implementation
   - Field equations (`equations`)
   - Linearized analysis (`linearized`)
   - Physical constraints (`constraints`)
   - Thermodynamic relations (`thermodynamics`)

3. **`rtrg.field_theory`** - MSRJD field theory formulation
   - Action principle (`msrjd_action`)
   - Propagator calculations (`propagators`)
   - Vertex extraction (`vertices`)
   - Feynman rules (`feynman_rules`)

4. **`rtrg.renormalization`** - RG analysis and calculations
   - One-loop calculations (`one_loop`)
   - Beta function extraction (`beta_functions`)
   - Fixed point analysis (`fixed_points`)
   - RG flow solutions (`flow`)

## Building and Running

### Setup
1. Install dependencies using `pip install -e .` (in development mode)
2. Alternatively, use `uv` for faster dependency resolution

### Running
- Execute the library: `relativistic-turbulence-rg` (defined in `pyproject.toml`)
- This will display version information and point to documentation

### Testing
- Unit tests are located in `tests/unit/`
- Integration tests in `tests/integration/`
- Benchmarks in `tests/benchmarks/`
- Run tests with: `pytest`

## Development Conventions

### Code Structure
- Modular design with clear separation of concerns
- Each module has a specific mathematical or physical purpose
- Fields are implemented as first-class objects with properties and constraints
- Tensor operations preserve Lorentz covariance and thermodynamic consistency

### Testing
- Tests are organized into unit, integration, and benchmark categories
- Uses `pytest` with custom markers (`unit`, `integration`, `physics`)
- Tests validate mathematical properties and physical constraints
- Includes validation of field properties, tensor symmetries, and evolution equations

### Documentation
- Inline docstrings in all modules
- Comprehensive module-level documentation
- Scientific references included in documentation
- Main documentation expected in `docs/` directory

## Qwen Added Memories
- The user prefers to use `uv` for managing the virtual environment for this project.
- The user prefers to run tests using the command `uv run pytest -v`.
