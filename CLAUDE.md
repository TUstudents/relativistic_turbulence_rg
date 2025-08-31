# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library for performing Renormalization Group analysis on relativistic Israel-Stewart equations to derive universal properties of turbulence. The project is in early development phase with planned implementation of field theory methods, symbolic computation, and numerical analysis for relativistic fluid dynamics.

## Development Commands

### Package Management
```bash
# Install dependencies and package in development mode
uv sync

# Run the main entry point
uv run relativistic-turbulence-rg

# Install package in development mode
uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test category
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/benchmarks/
```

### Code Quality
```bash
# Run linting (when configured)
uv run ruff check

# Format code (when configured) 
uv run ruff format

# Type checking (when configured)
uv run mypy src/
```

## Project Architecture

The project follows a modular scientific computing architecture organized around physics concepts:

### Core Structure
- `rtrg/`: Main package (planned structure from design docs)
  - `core/`: Fundamental mathematical objects (fields, tensors, parameters)
  - `israel_stewart/`: Relativistic hydrodynamics equations and constraints  
  - `field_theory/`: MSRJD action, propagators, vertices, and Feynman rules
  - `renormalization/`: RG flow equations and scaling analysis
- `src/relativistic_turbulence_rg/`: Current minimal implementation
- `tests/`: Test suite organized by type (unit, integration, benchmarks)
- `docs/`: Documentation split into theory, API reference, and tutorials
- `plan/`: Design documents and implementation roadmaps

### Key Dependencies
- Scientific computing: `numpy`, `scipy`, `sympy` for mathematical operations
- Performance: `numba` for JIT compilation of numerical kernels
- Visualization: `matplotlib` for plotting results
- Testing: `pytest` for comprehensive test coverage

### Development Notes
- Project uses `uv` for dependency management
- Entry point script available as `relativistic-turbulence-rg` command
- Heavy focus on symbolic computation for field theory calculations
- Designed for extensibility with plugin-style architecture for different physical models
- Implementation follows physics literature conventions for variable naming and mathematical notation
- plan/Israel-Stewart_Theory.md file contains the comprehensive theoretical foundation for the Israel-Stewart relativistic hydrodynamics
- plan/MSRJD_Formalism.md file provides the theoretical foundation for implementing the field theory approach to relativistic turbulence analysis
- plan/RG_Turbulence_Methods.md provides the methodological foundation for implementing the RG analysis
- License is CC BY-NC-SA 4.0

### Git Commit Messages
- **Short, technical summaries**: Max 50 characters for subject line
- **Concise descriptions**: Brief bullet points for multi-line commits
- **Professional tone**: Avoid flowery language or excessive detail
- **Action-oriented**: Start with imperative verbs (Fix, Add, Remove, Update)
- **Technical standard**: Follow conventional commit format when applicable
