# Relativistic Turbulence Renormalization Group

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A comprehensive research framework for investigating universal properties of relativistic turbulence through field-theoretic renormalization group analysis. This project implements the Israel-Stewart formalism of second-order relativistic hydrodynamics within the Martin-Siggia-Rose-Janssen-De Dominicis (MSRJD) field theory framework to derive universal scaling laws and transport coefficients in relativistic turbulent flows.

## Theoretical Foundation

### Israel-Stewart Relativistic Hydrodynamics
Second-order causal and stable formulation of relativistic fluid dynamics, extending the Navier-Stokes equations to relativistic regimes while maintaining causality and thermodynamic consistency. The theory introduces dissipative fluxes (shear stress π^μν, bulk viscous pressure Π, heat flux q^μ) with finite relaxation times.

### MSRJD Field Theory Framework  
The Martin-Siggia-Rose-Janssen-De Dominicis action provides a systematic field-theoretic approach to stochastic hydrodynamics, enabling the application of renormalization group techniques to derive universal properties independent of microscopic details.

### Renormalization Group Analysis
Systematic study of how transport coefficients and correlation functions change under coarse-graining, revealing universal scaling behaviors characteristic of turbulent flows in relativistic systems.

## Scientific Capabilities

### Relativistic Tensor Operations

- **Lorentz-Invariant Calculations**: Complete spacetime tensor algebra with metric-aware operations
- **Covariant Derivatives**: Christoffel symbol computations for curved spacetime
- **Index Management**: Automatic raising/lowering with metric tensor contractions
- **Tensor Symmetries**: Symmetrization, antisymmetrization, and trace operations
### Israel-Stewart Field Theory
- **Dissipative Fields**: Shear stress tensor π^μν, bulk viscous pressure Π, heat flux q^μ
- **Constraint Validation**: Orthogonality, tracelessness, and causality conditions  
- **Physical Parameters**: Transport coefficients with thermodynamic consistency
- **Relaxation Dynamics**: Finite-time evolution of dissipative fluxes

### MSRJD Action and Propagators
- **Field-Theoretic Action**: Complete MSRJD formulation for relativistic hydrodynamics
- **Propagator Theory**: Response and correlation functions in frequency-momentum space
- **Feynman Rules**: Systematic vertex calculations for perturbative analysis
- **Spectral Functions**: Mode analysis and pole structure of Green's functions

### Renormalization Group Tools
- **Scaling Analysis**: Identification of relevant and irrelevant operators
- **Flow Equations**: Beta functions for transport coefficient evolution
- **Universal Properties**: Scale-invariant ratios and critical exponents
- **Transport Coefficients**: Systematic calculation of viscosity and conductivity

## Research Applications

This framework enables systematic investigation of relativistic turbulence phenomena across multiple scales and physical systems:

- **Heavy-Ion Collision Physics**: Quark-gluon plasma evolution and transport properties in relativistic nuclear collisions
- **Relativistic Astrophysics**: Turbulent flows in neutron star mergers, accretion disks, and active galactic nuclei jets  
- **Fundamental Turbulence Theory**: Universal scaling laws and critical phenomena in relativistic fluid systems
- **Transport Phenomena**: Systematic calculation of shear viscosity, bulk viscosity, and thermal conductivity ratios
- **Cosmological Applications**: Primordial plasma dynamics and structure formation in the early universe

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
from rtrg.core import LorentzTensor, Metric, IndexStructure
from rtrg.israel_stewart import ISParameters, ShearStressField
from rtrg.field_theory import PropagatorCalculator

# Relativistic tensor operations in Minkowski spacetime  
metric = Metric()  # η_μν = diag(-1,1,1,1)
stress_components = np.diag([1.0, 0.3, 0.3, 0.3])  # Perfect fluid T^μν
indices = IndexStructure(['mu', 'nu'], ['contravariant', 'contravariant'], ['symmetric', 'symmetric'])
stress_tensor = LorentzTensor(stress_components, indices, metric)

# Lorentz-invariant trace: g_μν T^μν = -ρ + 3p  
trace = stress_tensor.trace()
print(f"Invariant trace: {trace}")  # Physics: -0.1

# Israel-Stewart dissipative hydrodynamics
shear_field = ShearStressField()
params = ISParameters(
    eta=0.2,      # Shear viscosity
    zeta=0.05,    # Bulk viscosity  
    tau_pi=1.0,   # Shear relaxation time
    cs=1/np.sqrt(3)  # Sound speed
)

# Field theory propagators and RG analysis
calc = PropagatorCalculator(params)
propagator = calc.calculate_retarded_propagator(shear_field, shear_field)
```

## Documentation

- **Theoretical Foundation**: Complete theoretical framework in `plan/` directory
  - `Israel-Stewart_Theory.md`: Relativistic hydrodynamics formalism
  - `MSRJD_Formalism.md`: Field-theoretic action and propagators  
  - `RG_Turbulence_Methods.md`: Renormalization group techniques
- **Physics Validation**: Comprehensive test suite ensuring theoretical correctness
- **API Reference**: Auto-generated from docstrings and type annotations
- **Testing**: Run `uv run pytest` for full validation suite

## Project Status

**Advanced Implementation** - Core theoretical framework complete with robust tensor algebra, field theory implementation, and comprehensive physics validation.

**Implemented Systems:**
- ✅ Lorentz-invariant tensor algebra with metric-aware operations
- ✅ Complete Israel-Stewart field definitions and constraint validation
- ✅ MSRJD field theory with propagators and Feynman rules
- ✅ Advanced propagator analysis with pole detection and spectral functions
- ✅ Comprehensive physics validation and testing framework

**Current Focus:** Renormalization group flow equations and scaling analysis.

See `plan/rtrg-task-list.md` for detailed development roadmap.

## References

### Israel-Stewart Theory
- Israel, W. & Stewart, J.M. "Transient relativistic thermodynamics and kinetic theory." Ann. Phys. **118**, 341 (1979)
- Müller, I. & Ruggeri, T. "Rational Extended Thermodynamics." Springer Tracts in Natural Philosophy (1998)
- Rezzolla, L. & Zanotti, O. "Relativistic Hydrodynamics." Oxford University Press (2013)

### MSRJD Field Theory and RG Methods
- Martin, P.C., Siggia, E.D. & Rose, H.A. "Statistical dynamics of classical systems." Phys. Rev. A **8**, 423 (1973)
- Janssen, H.K. "On a Lagrangian for classical field dynamics and renormalization group calculations of dynamical critical properties." Z. Phys. B **23**, 377 (1976)
- De Dominicis, C. "Techniques de renormalisation de la théorie des champs et dynamique des phénomènes critiques." J. Phys. Colloques **37**, C1-247 (1976)
- Forster, D., Nelson, D.R. & Stephen, M.J. "Large-distance and long-time properties of a randomly stirred fluid." Phys. Rev. A **16**, 732 (1977)

### Relativistic Turbulence and Transport
- Kovtun, P., Son, D.T. & Starinets, A.O. "Viscosity in strongly coupled quantum field theories from black hole physics." Phys. Rev. Lett. **94**, 111601 (2005)
- Policastro, G., Son, D.T. & Starinets, A.O. "The shear viscosity of strongly coupled N=4 supersymmetric Yang-Mills plasma." Phys. Rev. Lett. **87**, 081601 (2001)
- Romatschke, P. & Romatschke, U. "Relativistic Fluid Dynamics In and Out of Equilibrium." Cambridge University Press (2019)

## License

Licensed under [CC BY-NC-SA 4.0](LICENSE) - see LICENSE file for details.
