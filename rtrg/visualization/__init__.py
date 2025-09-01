"""Visualization tools for RG flow and Feynman diagrams"""

# Tensor diagrams (requires networkx)
try:
    from .tensor_diagrams import (
        plot_field_hierarchy,
        plot_tensor_operation,
        visualize_index_structure,
        visualize_tensor_network,
    )

    _tensor_diagrams_available = True
except ImportError:
    _tensor_diagrams_available = False

# Propagator plots (requires matplotlib)
try:
    from .propagator_plots import (
        PropagatorVisualizer,
        compare_propagator_types,
        plot_propagator_overview,
    )

    _propagator_plots_available = True
except ImportError:
    _propagator_plots_available = False

__all__ = []

if _tensor_diagrams_available:
    __all__.extend(
        [
            "visualize_tensor_network",
            "visualize_index_structure",
            "plot_tensor_operation",
            "plot_field_hierarchy",
        ]
    )

if _propagator_plots_available:
    __all__.extend(
        [
            "PropagatorVisualizer",
            "plot_propagator_overview",
            "compare_propagator_types",
        ]
    )
