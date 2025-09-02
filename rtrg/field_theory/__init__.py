"""MSRJD field theory formulation and Feynman rules"""

from .feynman_rules import FeynmanRules
from .msrjd_action import MSRJDAction

try:
    from .propagators import PropagatorCalculator
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import full propagators: {e}, using simplified version", stacklevel=2)
    try:
        from .propagators_simple import PropagatorCalculator
    except ImportError as e2:
        warnings.warn(f"Could not import simplified propagators: {e2}", stacklevel=2)
        PropagatorCalculator = None  # type: ignore[misc]
from .vertices import VertexExtractor

__all__ = ["MSRJDAction", "PropagatorCalculator", "VertexExtractor", "FeynmanRules"]
