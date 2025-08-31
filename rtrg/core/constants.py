"""
Physical constants, unit systems, and dimensional analysis for relativistic field theory.

This module provides fundamental physical constants, unit conversion utilities, and
dimensional analysis tools required for relativistic hydrodynamics and field theory
calculations. The implementation follows standard conventions from theoretical physics
with emphasis on natural unit systems commonly used in quantum field theory.

Natural Unit System:
    The primary system uses natural units where fundamental constants are unity:
    - Speed of light: c = 1
    - Reduced Planck constant: ℏ = 1  
    - Boltzmann constant: k_B = 1
    
    In this system, all quantities have dimensions of powers of mass [M^n].
    - Length: [L] = [M^{-1}]
    - Time: [T] = [M^{-1}]
    - Energy: [E] = [M]
    - Temperature: [Θ] = [M]

Metric Signature Convention:
    Uses the "mostly plus" metric signature (-,+,+,+) standard in particle physics
    and field theory. The line element is:
    ds² = -c²dt² + dx² + dy² + dz²

Dimensional Analysis Framework:
    Provides utilities for:
    - Computing dimensionless combinations of parameters
    - Validating dimensional consistency of equations
    - Converting between different unit systems
    - Extracting characteristic scales from parameter sets

Unit Conversion Support:
    Methods for converting between natural units and conventional systems (CGS, SI)
    with proper handling of relativistic factors and field theory normalizations.

References:
    - Peskin, M.E. & Schroeder, D.V. "An Introduction to Quantum Field Theory"
    - Weinberg, S. "The Quantum Theory of Fields" Vol. I
    - Srednicki, M. "Quantum Field Theory"
"""
import numpy as np
from typing import Dict, Any


class PhysicalConstants:
    """
    Fundamental physical constants in natural unit systems for field theory calculations.
    
    Provides access to fundamental constants with proper normalization for relativistic
    field theory and quantum field theory applications. The default system uses natural
    units where c = ℏ = k_B = 1, which is standard in theoretical physics.
    
    Natural Unit Advantages:
        - Simplifies equations by removing fundamental constants
        - Makes dimensional analysis transparent (everything in powers of mass)
        - Standard in quantum field theory and particle physics literature
        - Facilitates comparison with theoretical results
        
    Conversion Methods:
        Static methods provide conversion factors to CGS and SI systems for
        practical calculations and experimental comparison.
        
    Constants Available:
        c: Speed of light [dimensionless in natural units]
        ℏ: Reduced Planck constant [dimensionless in natural units]  
        k_B: Boltzmann constant [dimensionless in natural units]
        
    Examples:
        >>> # All fundamental constants are unity in natural units
        >>> assert PhysicalConstants.c == 1.0
        >>> assert PhysicalConstants.hbar == 1.0
        >>> assert PhysicalConstants.k_B == 1.0
        >>>
        >>> # Convert energy from natural units to ergs
        >>> energy_erg = PhysicalConstants.to_cgs(1.0, 'energy')  # 1 GeV → ergs
    """
    
    # Fundamental constants (in natural units)
    c = 1.0      # Speed of light
    hbar = 1.0   # Reduced Planck constant  
    k_B = 1.0    # Boltzmann constant
    
    # Spacetime signature: mostly plus convention (-,+,+,+)
    METRIC_SIGNATURE = (-1, 1, 1, 1)
    
    # Dimensional analysis base units
    LENGTH = 1  # [Length] 
    TIME = 1    # [Time]
    MASS = 1    # [Mass] 
    
    @classmethod
    def to_cgs(cls, quantity: float, dimension: str) -> float:
        """Convert from natural units to CGS"""
        conversions = {
            'length': 1.973e-11,  # cm (ℏc/GeV)
            'time': 6.582e-22,    # s (ℏ/GeV)
            'mass': 1.783e-24,    # g (GeV/c²)
            'energy': 1.602e-3,   # erg (GeV)
            'temperature': 1.160e13,  # K (GeV/k_B)
        }
        return quantity * conversions.get(dimension, 1.0)
    
    @classmethod  
    def dimensionless_parameters(cls, **kwargs) -> Dict[str, float]:
        """Compute dimensionless parameters for the theory"""
        params = {}
        
        # Reynolds number
        if all(k in kwargs for k in ['rho', 'v', 'L', 'eta']):
            params['Re'] = kwargs['rho'] * kwargs['v'] * kwargs['L'] / kwargs['eta']
            
        # Knudsen number 
        if all(k in kwargs for k in ['tau', 'c_s', 'L']):
            params['Kn'] = kwargs['tau'] * kwargs['c_s'] / kwargs['L']
            
        # Mach number
        if all(k in kwargs for k in ['v', 'c_s']):
            params['Ma'] = kwargs['v'] / kwargs['c_s']
            
        return params


class UnitSystem:
    """Handle different unit systems and conversions"""
    
    def __init__(self, system: str = 'natural'):
        """Initialize unit system
        
        Args:
            system: 'natural', 'cgs', or 'si'
        """
        self.system = system
        self.conversions = self._setup_conversions()
    
    def _setup_conversions(self) -> Dict[str, Dict[str, float]]:
        """Setup unit conversion factors"""
        return {
            'natural': {
                'length': 1.0,
                'time': 1.0, 
                'mass': 1.0,
                'energy': 1.0
            },
            'cgs': {
                'length': 1.973e-11,  # cm
                'time': 6.582e-22,    # s
                'mass': 1.783e-24,    # g  
                'energy': 1.602e-3    # erg
            }
        }
    
    def convert(self, value: float, dimension: str, 
                from_system: str = 'natural') -> float:
        """Convert between unit systems"""
        if from_system == self.system:
            return value
            
        # Convert via natural units
        if from_system != 'natural':
            value = value / self.conversions[from_system][dimension]
        if self.system != 'natural':
            value = value * self.conversions[self.system][dimension]
            
        return value