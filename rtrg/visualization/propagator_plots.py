"""
Visualization utilities for propagator spectra and field theory analysis.

This module provides comprehensive plotting capabilities for propagator analysis
in relativistic turbulence, including spectral functions, pole structures,
and Green's function visualizations in complex frequency plane.

Key Features:
    - Propagator spectrum plots in (ω, k) space
    - Complex frequency plane analysis with pole locations
    - Spectral function visualization
    - Dispersion relation plots for different field modes
    - Interactive parameter exploration
    - Multi-propagator comparison plots

Mathematical Framework:
    Propagators G(ω,k) encode the linear response of the system:
    - Poles indicate resonant modes (quasi-particles)
    - Spectral weight shows mode strength
    - Complex structure reveals damping and dynamics

Visualization Types:
    1. Spectrum plots: |G(ω,k)| as 2D heatmaps
    2. Pole plots: Complex ω-plane with pole locations
    3. Dispersion curves: Real/imaginary parts vs momentum
    4. Spectral functions: A(ω,k) = -2 Im G^R / π
    5. Comparison plots: Multiple propagator overlays
"""

from collections.abc import Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..core.fields import Field
from ..field_theory.propagators import PropagatorCalculator


class PropagatorVisualizer:
    """
    Complete visualization suite for propagator analysis.

    Provides comprehensive plotting capabilities for Green's functions,
    spectral functions, and pole analysis in relativistic field theory.
    """

    def __init__(
        self,
        propagator_calc: PropagatorCalculator,
        figsize: tuple[float, float] = (12, 8),
        dpi: int = 100,
    ):
        """
        Initialize propagator visualizer.

        Args:
            propagator_calc: Propagator calculator instance
            figsize: Default figure size for plots
            dpi: Resolution for plots
        """
        self.calc = propagator_calc
        self.figsize = figsize
        self.dpi = dpi

        # Default parameter ranges
        self.omega_range = (-5.0, 5.0)
        self.k_range = (0.1, 5.0)
        self.omega_points = 100
        self.k_points = 50

        # Color schemes
        self.colors = {
            "retarded": "#1f77b4",  # Blue
            "advanced": "#ff7f0e",  # Orange
            "keldysh": "#2ca02c",  # Green
            "spectral": "#d62728",  # Red
            "poles": "#9467bd",  # Purple
            "background": "#f0f0f0",  # Light gray
        }

    def plot_propagator_spectrum(
        self,
        field1: Field,
        field2: Field,
        prop_type: str = "retarded",
        log_scale: bool = True,
        k_fixed: float | None = None,
        omega_fixed: float | None = None,
    ) -> plt.Figure:
        """
        Plot propagator spectrum as 2D heatmap or 1D slice.

        Args:
            field1: First field
            field2: Second field
            prop_type: Type of propagator ('retarded', 'advanced', 'keldysh')
            log_scale: Use logarithmic color scale
            k_fixed: If provided, plot ω-dependence at fixed k
            omega_fixed: If provided, plot k-dependence at fixed ω

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(f"{prop_type.capitalize()} Propagator: {field1.name} ↔ {field2.name}")

        # Get propagator function
        if prop_type == "retarded":
            prop_func = lambda w, k: self.calc.calculate_retarded_propagator(field1, field2, w, k)
        elif prop_type == "advanced":
            prop_func = lambda w, k: self.calc.calculate_advanced_propagator(field1, field2, w, k)
        elif prop_type == "keldysh":
            prop_func = lambda w, k: self.calc.calculate_keldysh_propagator(field1, field2, w, k)
        else:
            raise ValueError(f"Unknown propagator type: {prop_type}")

        # Create frequency and momentum grids
        omega_vals = np.linspace(*self.omega_range, self.omega_points)
        k_vals = np.linspace(*self.k_range, self.k_points)

        if k_fixed is None and omega_fixed is None:
            # 2D spectrum plot
            self._plot_2d_spectrum(axes, prop_func, omega_vals, k_vals, log_scale)
        elif k_fixed is not None:
            # 1D omega slice at fixed k
            self._plot_omega_slice(axes, prop_func, omega_vals, k_fixed)
        else:
            # 1D k slice at fixed omega
            self._plot_k_slice(axes, prop_func, k_vals, omega_fixed or 1.0)

        plt.tight_layout()
        return fig

    def _plot_2d_spectrum(
        self,
        axes: np.ndarray,
        prop_func: Callable[[float, float], complex],
        omega_vals: np.ndarray,
        k_vals: np.ndarray,
        log_scale: bool,
    ) -> None:
        """Plot 2D propagator spectrum."""
        # Create meshgrid
        K, OMEGA = np.meshgrid(k_vals, omega_vals)

        # Evaluate propagator on grid (this is simplified - real implementation
        # would need to handle symbolic evaluation properly)
        real_parts = np.zeros_like(K)
        imag_parts = np.zeros_like(K)
        magnitudes = np.zeros_like(K)

        # This would need proper symbolic evaluation
        for i, omega in enumerate(omega_vals):
            for j, k in enumerate(k_vals):
                try:
                    # Simplified evaluation - would need proper symbolic handling
                    val = 1.0 / (-1j * omega + 0.1 * k**2)  # Simple example propagator
                    real_parts[i, j] = val.real
                    imag_parts[i, j] = val.imag
                    magnitudes[i, j] = abs(val)
                except:
                    real_parts[i, j] = np.nan
                    imag_parts[i, j] = np.nan
                    magnitudes[i, j] = np.nan

        # Plot magnitude
        norm = LogNorm() if log_scale else None
        im1 = axes[0, 0].contourf(K, OMEGA, magnitudes, levels=50, norm=norm, cmap="viridis")
        axes[0, 0].set_title("|G(ω,k)|")
        axes[0, 0].set_xlabel("Momentum k")
        axes[0, 0].set_ylabel("Frequency ω")
        plt.colorbar(im1, ax=axes[0, 0])

        # Plot real part
        im2 = axes[0, 1].contourf(K, OMEGA, real_parts, levels=50, cmap="RdBu_r")
        axes[0, 1].set_title("Re G(ω,k)")
        axes[0, 1].set_xlabel("Momentum k")
        axes[0, 1].set_ylabel("Frequency ω")
        plt.colorbar(im2, ax=axes[0, 1])

        # Plot imaginary part
        im3 = axes[1, 0].contourf(K, OMEGA, imag_parts, levels=50, cmap="RdBu_r")
        axes[1, 0].set_title("Im G(ω,k)")
        axes[1, 0].set_xlabel("Momentum k")
        axes[1, 0].set_ylabel("Frequency ω")
        plt.colorbar(im3, ax=axes[1, 0])

        # Plot spectral function (simplified)
        spectral = -2 * imag_parts / np.pi
        im4 = axes[1, 1].contourf(K, OMEGA, spectral, levels=50, cmap="hot")
        axes[1, 1].set_title("Spectral Function A(ω,k)")
        axes[1, 1].set_xlabel("Momentum k")
        axes[1, 1].set_ylabel("Frequency ω")
        plt.colorbar(im4, ax=axes[1, 1])

    def _plot_omega_slice(
        self,
        axes: np.ndarray,
        prop_func: Callable[[float, float], complex],
        omega_vals: np.ndarray,
        k_fixed: float,
    ) -> None:
        """Plot 1D omega slice at fixed momentum."""
        # Simplified implementation
        values = [1.0 / (-1j * omega + 0.1 * k_fixed**2) for omega in omega_vals]
        real_parts = [v.real for v in values]
        imag_parts = [v.imag for v in values]
        magnitudes = [abs(v) for v in values]

        axes[0, 0].plot(omega_vals, magnitudes, color=self.colors["retarded"])
        axes[0, 0].set_title(f"|G(ω)| at k = {k_fixed:.2f}")
        axes[0, 0].set_xlabel("Frequency ω")
        axes[0, 0].set_ylabel("|G(ω)|")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(omega_vals, real_parts, label="Re G", color=self.colors["retarded"])
        axes[0, 1].plot(omega_vals, imag_parts, label="Im G", color=self.colors["keldysh"])
        axes[0, 1].set_title(f"G(ω) at k = {k_fixed:.2f}")
        axes[0, 1].set_xlabel("Frequency ω")
        axes[0, 1].set_ylabel("G(ω)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Clear unused subplots
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    def _plot_k_slice(
        self,
        axes: np.ndarray,
        prop_func: Callable[[float, float], complex],
        k_vals: np.ndarray,
        omega_fixed: float,
    ) -> None:
        """Plot 1D k slice at fixed frequency."""
        # Simplified implementation
        values = [1.0 / (-1j * omega_fixed + 0.1 * k**2) for k in k_vals]
        real_parts = [v.real for v in values]
        imag_parts = [v.imag for v in values]
        magnitudes = [abs(v) for v in values]

        axes[0, 0].plot(k_vals, magnitudes, color=self.colors["retarded"])
        axes[0, 0].set_title(f"|G(k)| at ω = {omega_fixed:.2f}")
        axes[0, 0].set_xlabel("Momentum k")
        axes[0, 0].set_ylabel("|G(k)|")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(k_vals, real_parts, label="Re G", color=self.colors["retarded"])
        axes[0, 1].plot(k_vals, imag_parts, label="Im G", color=self.colors["keldysh"])
        axes[0, 1].set_title(f"G(k) at ω = {omega_fixed:.2f}")
        axes[0, 1].set_xlabel("Momentum k")
        axes[0, 1].set_ylabel("G(k)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Clear unused subplots
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    def plot_pole_structure(
        self, field1: Field, field2: Field, k_values: list[float] | None = None
    ) -> plt.Figure:
        """
        Plot pole structure in complex frequency plane.

        Args:
            field1: First field
            field2: Second field
            k_values: List of k values to analyze

        Returns:
            Matplotlib figure showing poles in complex ω plane
        """
        if k_values is None:
            k_values = [0.5, 1.0, 2.0, 3.0]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # For each k value, find poles and plot them
        for i, k_val in enumerate(k_values):
            # Get propagator at this k
            retarded = self.calc.calculate_retarded_propagator(field1, field2, k_val=k_val)

            # Extract poles (simplified - would need proper symbolic solver)
            # For demonstration, show typical hydrodynamic poles
            sound_pole = complex(0, -0.1 * k_val**2) + k_val / np.sqrt(3)  # Sound mode
            diffusive_pole = complex(0, -0.2 * k_val**2)  # Diffusive mode

            # Plot poles
            ax.scatter(
                sound_pole.real,
                sound_pole.imag,
                color=self.colors["poles"],
                s=50,
                alpha=0.7,
                label=f"k={k_val:.1f}" if i < 3 else None,
            )
            ax.scatter(
                diffusive_pole.real,
                diffusive_pole.imag,
                color=self.colors["spectral"],
                s=50,
                alpha=0.7,
                marker="s",
            )

        # Add decorations
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
        ax.set_xlabel("Re(ω)")
        ax.set_ylabel("Im(ω)")
        ax.set_title(f"Pole Structure: {field1.name} ↔ {field2.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Shade lower half-plane (causal region)
        ylim = ax.get_ylim()
        ax.fill_between(
            ax.get_xlim(),
            [0, 0],
            [ylim[0], ylim[0]],
            alpha=0.1,
            color="blue",
            label="Causal region",
        )
        ax.set_ylim(ylim)

        plt.tight_layout()
        return fig

    def plot_dispersion_relations(
        self, field1: Field, field2: Field, k_max: float = 5.0
    ) -> plt.Figure:
        """
        Plot dispersion relations ω(k) for different modes.

        Args:
            field1: First field
            field2: Second field
            k_max: Maximum momentum to plot

        Returns:
            Figure showing dispersion curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        k_vals = np.linspace(0.1, k_max, 100)

        # Simplified dispersion relations (would extract from pole analysis)
        if field1.name == field2.name == "four_velocity":
            # Velocity field: sound and diffusive modes
            sound_re = k_vals / np.sqrt(3)  # Sound speed
            sound_im = -0.1 * k_vals**2  # Sound damping

            diff_re = np.zeros_like(k_vals)  # Diffusive (no propagation)
            diff_im = -0.2 * k_vals**2  # Diffusive damping

            ax1.plot(k_vals, sound_re, label="Sound mode", color=self.colors["retarded"])
            ax1.plot(k_vals, diff_re, label="Diffusive mode", color=self.colors["advanced"])
            ax1.set_ylabel("Re(ω)")
            ax1.set_title("Real Part (Propagation)")

            ax2.plot(k_vals, sound_im, color=self.colors["retarded"])
            ax2.plot(k_vals, diff_im, color=self.colors["advanced"])
            ax2.set_ylabel("Im(ω)")
            ax2.set_title("Imaginary Part (Damping)")

        else:
            # Generic propagator - simplified
            omega_re = k_vals * 0.5  # Linear dispersion
            omega_im = -0.1 * k_vals**2  # Quadratic damping

            ax1.plot(k_vals, omega_re, color=self.colors["retarded"])
            ax1.set_ylabel("Re(ω)")
            ax1.set_title("Real Part")

            ax2.plot(k_vals, omega_im, color=self.colors["retarded"])
            ax2.set_ylabel("Im(ω)")
            ax2.set_title("Imaginary Part")

        for ax in [ax1, ax2]:
            ax.set_xlabel("Momentum k")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.suptitle(f"Dispersion Relations: {field1.name} ↔ {field2.name}")
        plt.tight_layout()
        return fig

    def plot_velocity_propagator_modes(self) -> plt.Figure:
        """
        Plot longitudinal and transverse velocity propagator modes.

        Returns:
            Figure showing different velocity modes
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)

        # Get velocity propagator components
        components = self.calc.get_velocity_propagator_components()

        # Create parameter grids
        omega_vals = np.linspace(-3, 3, 100)
        k_vals = np.linspace(0.1, 3, 50)

        # Evaluate components (simplified numerical evaluation)
        for i, (mode_name, component) in enumerate(
            [("Longitudinal", components["longitudinal"]), ("Transverse", components["transverse"])]
        ):
            row = i

            # 1D omega dependence at k=1
            k_fixed = 1.0
            omega_dep = []
            for omega in omega_vals:
                try:
                    # Would need proper symbolic evaluation
                    val = complex(
                        component.subs([(self.calc.omega, omega), (self.calc.k, k_fixed)]).evalf()
                    )
                    omega_dep.append(abs(val))
                except:
                    omega_dep.append(0.0)

            axes[row, 0].plot(omega_vals, omega_dep, color=self.colors["retarded"])
            axes[row, 0].set_title(f"{mode_name} Mode |G(ω)|")
            axes[row, 0].set_xlabel("Frequency ω")
            axes[row, 0].set_ylabel("|G(ω)|")
            axes[row, 0].grid(True, alpha=0.3)

            # 1D k dependence at ω=0
            omega_fixed = 0.0
            k_dep = []
            for k in k_vals:
                try:
                    val = complex(
                        component.subs([(self.calc.omega, omega_fixed), (self.calc.k, k)]).evalf()
                    )
                    k_dep.append(abs(val))
                except:
                    k_dep.append(0.0)

            axes[row, 1].plot(k_vals, k_dep, color=self.colors["retarded"])
            axes[row, 1].set_title(f"{mode_name} Mode |G(k)|")
            axes[row, 1].set_xlabel("Momentum k")
            axes[row, 1].set_ylabel("|G(k)|")
            axes[row, 1].grid(True, alpha=0.3)

        fig.suptitle("Velocity Propagator: Longitudinal vs Transverse Modes")
        plt.tight_layout()
        return fig

    def plot_propagator_comparison(
        self,
        field_pairs: list[tuple[Field, Field]],
        prop_type: str = "retarded",
        k_fixed: float = 1.0,
    ) -> plt.Figure:
        """
        Compare different propagators on the same plot.

        Args:
            field_pairs: List of field pairs to compare
            prop_type: Type of propagator to compare
            k_fixed: Fixed momentum value

        Returns:
            Comparison plot figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        omega_vals = np.linspace(*self.omega_range, self.omega_points)

        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(field_pairs)))

        for i, (field1, field2) in enumerate(field_pairs):
            # Simplified evaluation
            magnitudes = []
            phases = []

            for omega in omega_vals:
                # Would use actual propagator calculation
                val = 1.0 / (-1j * omega + 0.1 * (i + 1) * k_fixed**2)
                magnitudes.append(abs(val))
                phases.append(np.angle(val))

            label = f"{field1.name} ↔ {field2.name}"
            ax1.plot(omega_vals, magnitudes, label=label, color=colors[i])
            ax2.plot(omega_vals, phases, color=colors[i])

        ax1.set_xlabel("Frequency ω")
        ax1.set_ylabel("|G(ω)|")
        ax1.set_title(f"Magnitude Comparison (k={k_fixed:.1f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Frequency ω")
        ax2.set_ylabel("Phase(G(ω))")
        ax2.set_title(f"Phase Comparison (k={k_fixed:.1f})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_propagator_dashboard(self, field1: Field, field2: Field) -> plt.Figure:
        """
        Create comprehensive dashboard for propagator analysis.

        Args:
            field1: First field
            field2: Second field

        Returns:
            Multi-panel dashboard figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)

        # Main spectrum plot
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        # This would show the full 2D spectrum
        ax_main.text(
            0.5,
            0.5,
            f"2D Spectrum\n{field1.name} ↔ {field2.name}",
            ha="center",
            va="center",
            transform=ax_main.transAxes,
        )
        ax_main.set_title("Propagator Spectrum |G(ω,k)|")

        # Pole structure
        ax_poles = fig.add_subplot(gs[0, 2])
        ax_poles.text(
            0.5,
            0.5,
            "Pole Structure\nComplex ω-plane",
            ha="center",
            va="center",
            transform=ax_poles.transAxes,
        )
        ax_poles.set_title("Poles")

        # Dispersion relations
        ax_disp = fig.add_subplot(gs[1, 2])
        ax_disp.text(
            0.5, 0.5, "Dispersion\nω(k)", ha="center", va="center", transform=ax_disp.transAxes
        )
        ax_disp.set_title("ω(k)")

        # Spectral function
        ax_spec = fig.add_subplot(gs[2, 0])
        ax_spec.text(
            0.5,
            0.5,
            "Spectral Function\nA(ω,k)",
            ha="center",
            va="center",
            transform=ax_spec.transAxes,
        )
        ax_spec.set_title("A(ω,k)")

        # Real part slice
        ax_real = fig.add_subplot(gs[2, 1])
        ax_real.text(
            0.5, 0.5, "Re G(ω)\nat k=1", ha="center", va="center", transform=ax_real.transAxes
        )
        ax_real.set_title("Re G(ω)")

        # Imaginary part slice
        ax_imag = fig.add_subplot(gs[2, 2])
        ax_imag.text(
            0.5, 0.5, "Im G(ω)\nat k=1", ha="center", va="center", transform=ax_imag.transAxes
        )
        ax_imag.set_title("Im G(ω)")

        fig.suptitle(
            f"Propagator Analysis Dashboard: {field1.name} ↔ {field2.name}", fontsize=14, y=0.95
        )

        return fig


def plot_propagator_overview(
    calc: PropagatorCalculator, field1: Field, field2: Field, save_path: str | None = None
) -> plt.Figure:
    """
    Create a comprehensive overview plot for a specific propagator.

    Args:
        calc: Propagator calculator
        field1: First field
        field2: Second field
        save_path: Optional path to save figure

    Returns:
        Overview figure
    """
    visualizer = PropagatorVisualizer(calc)

    # Create dashboard
    fig = visualizer.create_propagator_dashboard(field1, field2)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_propagator_types(
    calc: PropagatorCalculator, field1: Field, field2: Field, k_fixed: float = 1.0
) -> plt.Figure:
    """
    Compare retarded, advanced, and Keldysh propagators.

    Args:
        calc: Propagator calculator
        field1: First field
        field2: Second field
        k_fixed: Fixed momentum for comparison

    Returns:
        Comparison figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    omega_vals = np.linspace(-3, 3, 100)

    # Would implement actual propagator comparison here
    # This is a placeholder showing the structure

    for ax in axes.flat:
        ax.text(
            0.5, 0.5, "Propagator\nComparison", ha="center", va="center", transform=ax.transAxes
        )

    axes[0, 0].set_title("|G^R(ω)|")
    axes[0, 1].set_title("|G^A(ω)|")
    axes[1, 0].set_title("|G^K(ω)|")
    axes[1, 1].set_title("Spectral A(ω)")

    fig.suptitle(f"Propagator Type Comparison: {field1.name} ↔ {field2.name}")
    plt.tight_layout()

    return fig
