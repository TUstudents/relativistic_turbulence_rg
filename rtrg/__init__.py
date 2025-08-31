"""
Relativistic Turbulence Renormalization Group Analysis Library
==============================================================

Professional implementation of renormalization group methods for relativistic
Israel-Stewart hydrodynamics, enabling systematic derivation of universal
scaling properties in relativistic turbulent flows.

Theoretical Framework:
    This library implements the complete mathematical framework for applying 
    renormalization group techniques to the Israel-Stewart equations of 
    relativistic viscous hydrodynamics. The approach resolves fundamental 
    issues in turbulence theory by utilizing the causal structure and 
    Lorentz covariance of relativistic field theory.

Core Capabilities:
    - Complete tensor algebra system for Lorentz covariant calculations
    - Field definitions for all Israel-Stewart theory variables
    - MSRJD path integral formulation for stochastic hydrodynamics  
    - One-loop renormalization group calculations
    - Fixed point analysis and universal exponent extraction
    - Systematic non-relativistic limit procedures

Mathematical Infrastructure:
    Built on rigorous field-theoretic foundations with automatic handling
    of tensor indices, constraint enforcement, and dimensional analysis.
    All calculations preserve Lorentz covariance and thermodynamic consistency.

Scientific Applications:
    - Theoretical turbulence research
    - Relativistic fluid dynamics modeling
    - Field theory applications to hydrodynamics
    - Universal scaling analysis
    - Heavy-ion collision physics
    - Astrophysical fluid dynamics

References:
    - Israel, W. & Stewart, J.M. Ann. Phys. 118, 341 (1979)
    - Forster, D. et al. Phys. Rev. A 16, 732 (1977) 
    - Kovtun, P. et al. JHEP 10, 064 (2011)
"""

__version__ = "0.1.0"
__author__ = "Aristos"

# Core modules
from . import core
from . import israel_stewart  
from . import field_theory
from . import renormalization

def main() -> None:
    """
    Display library information and version details.
    
    Provides basic identification of the library version and directs
    users to documentation resources for detailed usage information.
    """
    print(f"Relativistic Turbulence RG Library v{__version__}")
    print("Professional implementation of RG methods for relativistic hydrodynamics.")
    print("Documentation: See docs/ directory and inline docstrings.")