# Relativistic Turbulence Renormalization Group

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Professional implementation of renormalization group methods for relativistic Israel-Stewart hydrodynamics, enabling systematic derivation of universal scaling properties in relativistic turbulent flows.

## Features

- **Lorentz Tensor Algebra**: Complete tensor system with automatic index management
- **Israel-Stewart Fields**: Field definitions with constraint validation 
- **Parameter Management**: Physical parameter validation with causality checks
- **Natural Units**: Full support for natural unit systems (c =  = k_B = 1)
- **Extensible Architecture**: Modular design for field theory calculations

## Installation

### Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Basic Installation
```bash
# Clone repository
git clone https://github.com/TUstudents/relativistic_turbulence_rg.git
cd relativistic_turbulence_rg

# Install dependencies and package
uv sync
uv pip install -e .

# Verify installation
uv run relativistic-turbulence-rg
```

### Development Installation
```bash
# Install with development tools
uv sync --extra dev

# Install with testing tools
uv sync --extra test

# Install everything
uv sync --extra all
```

### Available Commands
```bash
# Run tests
uv run pytest

# Code formatting and linting
uv run ruff check
uv run ruff format

# Type checking
uv run mypy rtrg/
```

## Quick Start

```python
import numpy as np
from rtrg.core import LorentzTensor, EnergyDensityField, ISParameters

# Create tensor operations
metric = Metric()
tensor = LorentzTensor(np.eye(4), metric=metric)

# Define fields with constraints
rho_field = EnergyDensityField()

# Setup physical parameters
params = ISParameters(
    eta=1.0, zeta=0.1, kappa=1.0,
    tau_pi=1.0, tau_Pi=0.5, tau_q=1.0,
    cs=0.577, temperature=1.0
)
```

## Documentation

- **Theoretical Foundation**: See `plan/` directory for detailed theory
- **API Reference**: [TODO: Generate API docs]
- **Tutorials**: [TODO: Add tutorial notebooks]
- **Testing**: Run `uv run pytest` for test suite

## Project Status

**Early Development** - Task 1.1 (Project Setup and Core Architecture) completed.

See `plan/rtrg-task-list.md` for development roadmap.

## References

- Israel, W. & Stewart, J.M. Ann. Phys. 118, 341 (1979)
- Forster, D. et al. Phys. Rev. A 16, 732 (1977)
- Kovtun, P. et al. JHEP 10, 064 (2011)

## License

Licensed under [CC BY-NC-SA 4.0](LICENSE) - see LICENSE file for details.