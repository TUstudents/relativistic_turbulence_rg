"""
Tensor network visualization utilities for relativistic field theory.

This module provides tools for visualizing tensor contractions, index structures,
and mathematical relationships in relativistic hydrodynamics calculations. The
visualizations help understand complex tensor operations and validate calculations.

Features:
    - Tensor network diagrams showing contraction patterns
    - Index structure visualization for debugging
    - Mathematical expression rendering
    - Interactive plots for tensor operations

Mathematical Context:
    Tensor networks represent multilinear operations as graphs where:
    - Nodes represent tensors
    - Edges represent indices
    - Connected edges show contractions (Einstein summation)
    - Edge styles indicate covariant/contravariant types
"""

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..core.registry_base import AbstractFieldRegistry
from ..core.registry_factory import create_registry_for_context
from ..core.tensors import LorentzTensor


def visualize_tensor_network(
    tensors: dict[str, LorentzTensor],
    contractions: list[tuple[str, int, str, int]],
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize tensor contraction network as a graph diagram.

    Creates a visual representation of tensor operations showing how indices
    are contracted between tensors. Useful for understanding complex
    relativistic calculations and debugging tensor operations.

    Args:
        tensors: Dictionary mapping tensor names to LorentzTensor objects
        contractions: List of contractions as (tensor1, index1, tensor2, index2) tuples
        figsize: Figure size for matplotlib plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object containing the visualization

    Examples:
        >>> # Visualize T^μν contraction with u_μ
        >>> tensors = {
        ...     'T': stress_energy_tensor,    # T^μν
        ...     'u': four_velocity           # u_μ
        ... }
        >>> contractions = [('T', 0, 'u', 0)]  # Contract μ indices
        >>> fig = visualize_tensor_network(tensors, contractions)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create networkx graph
    G: nx.Graph = nx.Graph()

    # Add tensor nodes
    tensor_positions = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        # Position tensors in a circle
        angle = 2 * np.pi * i / len(tensors)
        x, y = 3 * np.cos(angle), 3 * np.sin(angle)
        tensor_positions[name] = (x, y)

        G.add_node(name, pos=(x, y), rank=tensor.rank, type="tensor")

        # Add index nodes for each tensor
        for j, idx_name in enumerate(tensor.indices.names):
            idx_node = f"{name}_{idx_name}_{j}"
            # Position indices around tensor
            idx_angle = angle + (j - tensor.rank / 2) * 0.3
            idx_x = x + 0.8 * np.cos(idx_angle)
            idx_y = y + 0.8 * np.sin(idx_angle)

            G.add_node(
                idx_node,
                pos=(idx_x, idx_y),
                index_type=tensor.indices.types[j],
                symmetry=tensor.indices.symmetries[j],
                type="index",
            )

            # Connect tensor to its indices
            G.add_edge(name, idx_node, type="tensor_index")

    # Add contraction edges
    for tensor1, idx1, tensor2, idx2 in contractions:
        if tensor1 not in tensors or tensor2 not in tensors:
            continue

        node1 = f"{tensor1}_{tensors[tensor1].indices.names[idx1]}_{idx1}"
        node2 = f"{tensor2}_{tensors[tensor2].indices.names[idx2]}_{idx2}"

        if node1 in G.nodes and node2 in G.nodes:
            G.add_edge(node1, node2, type="contraction")

    # Draw the graph
    pos = nx.get_node_attributes(G, "pos")

    # Draw tensor nodes (large squares)
    tensor_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "tensor"]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tensor_nodes,
        node_shape="s",
        node_size=2000,
        node_color="lightblue",
        alpha=0.8,
    )

    # Draw index nodes (small circles)
    index_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "index"]

    # Color code by index type
    covariant_nodes = [n for n in index_nodes if G.nodes[n]["index_type"] == "covariant"]
    contravariant_nodes = [n for n in index_nodes if G.nodes[n]["index_type"] == "contravariant"]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=covariant_nodes,
        node_shape="o",
        node_size=200,
        node_color="red",
        alpha=0.7,
        label="Covariant",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=contravariant_nodes,
        node_shape="^",
        node_size=200,
        node_color="blue",
        alpha=0.7,
        label="Contravariant",
    )

    # Draw edges
    tensor_index_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "tensor_index"]
    contraction_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "contraction"]

    nx.draw_networkx_edges(
        G, pos, edgelist=tensor_index_edges, edge_color="gray", alpha=0.6, width=1
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=contraction_edges, edge_color="red", alpha=0.8, width=3, style="dashed"
    )

    # Add labels
    tensor_labels = {n: n for n in tensor_nodes}
    nx.draw_networkx_labels(G, pos, tensor_labels, font_size=12, font_weight="bold")

    # Index labels (smaller)
    index_labels = {}
    for node in index_nodes:
        parts = node.split("_")
        if len(parts) >= 2:
            index_labels[node] = parts[1]  # Just the index name

    nx.draw_networkx_labels(G, pos, index_labels, font_size=8, font_color="black")

    ax.set_title("Tensor Contraction Network", fontsize=16, fontweight="bold")
    ax.legend()
    ax.axis("equal")
    ax.axis("off")

    # Add explanatory text
    explanation = (
        "Tensor Network Visualization:\n"
        "• Blue squares: Tensors\n"
        "• Red circles: Covariant indices (lower)\n"
        "• Blue triangles: Contravariant indices (upper)\n"
        "• Dashed red lines: Contractions (Einstein summation)"
    )
    ax.text(
        0.02,
        0.98,
        explanation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def visualize_index_structure(
    tensor: LorentzTensor, figsize: tuple[int, int] = (10, 6), save_path: str | None = None
) -> plt.Figure:
    """
    Visualize the index structure of a single tensor.

    Creates a detailed view of tensor indices showing their types, symmetries,
    and relationships. Useful for debugging tensor operations and understanding
    the mathematical structure.

    Args:
        tensor: LorentzTensor to visualize
        figsize: Figure size for matplotlib plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object

    Examples:
        >>> # Visualize stress-energy tensor T^μν structure
        >>> fig = visualize_index_structure(stress_energy_tensor)
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Index diagram
    indices = tensor.indices

    # Draw tensor as central box
    tensor_box = patches.Rectangle(
        (0.4, 0.4), 0.2, 0.2, facecolor="lightblue", edgecolor="black", linewidth=2
    )
    ax1.add_patch(tensor_box)
    ax1.text(0.5, 0.5, "T", ha="center", va="center", fontsize=16, fontweight="bold")

    # Draw indices around the tensor
    positions = []
    for i, (name, idx_type, symmetry) in enumerate(
        zip(indices.names, indices.types, indices.symmetries)
    ):
        # Position indices around the tensor
        angle = 2 * np.pi * i / len(indices.names)
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        positions.append((x, y))

        # Color by type
        color = "red" if idx_type == "covariant" else "blue"
        marker = "v" if idx_type == "covariant" else "^"

        ax1.scatter(x, y, s=300, c=color, marker=marker, alpha=0.7, edgecolors="black")

        # Label with index name
        offset_x = 0.1 * np.cos(angle)
        offset_y = 0.1 * np.sin(angle)
        ax1.text(
            x + offset_x,
            y + offset_y,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Draw line from tensor to index
        ax1.plot([0.5, x], [0.5, y], "k-", alpha=0.5, linewidth=2)

        # Add symmetry indicator
        if symmetry != "none":
            symbol = "S" if symmetry == "symmetric" else "A"
            ax1.text(
                x,
                y - 0.03,
                symbol,
                ha="center",
                va="top",
                fontsize=8,
                color="green",
                fontweight="bold",
            )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_title(f"Index Structure (Rank {tensor.rank})", fontweight="bold")
    ax1.axis("off")

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="Contravariant (upper)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Covariant (lower)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="green",
            markersize=8,
            label="S: Symmetric, A: Antisymmetric",
            linestyle="None",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    # Right plot: Component matrix visualization (for rank ≤ 2)
    if tensor.rank <= 2:
        if tensor.rank == 0:
            # Scalar
            ax2.text(
                0.5,
                0.5,
                f"Scalar value: {tensor.components:.3f}",
                ha="center",
                va="center",
                fontsize=12,
            )
        elif tensor.rank == 1:
            # Vector
            components = (
                np.real(tensor.components)
                if np.isreal(tensor.components).all()
                else np.abs(tensor.components)
            )
            ax2.bar(range(len(components)), components)
            ax2.set_xlabel("Component index")
            ax2.set_ylabel("Value")
            ax2.set_title("Vector Components")
        else:
            # Matrix
            components = (
                np.real(tensor.components)
                if np.isreal(tensor.components).all()
                else np.abs(tensor.components)
            )
            im = ax2.imshow(components, cmap="RdBu_r", aspect="equal")
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Tensor Components")
            ax2.set_xlabel("Index 1")
            ax2.set_ylabel("Index 0")
    else:
        ax2.text(
            0.5,
            0.5,
            f"Rank {tensor.rank} tensor\n(too high to visualize)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.set_title("High-Rank Tensor")

    ax2.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_tensor_operation(
    operation_name: str,
    input_tensors: dict[str, LorentzTensor],
    result_tensor: LorentzTensor,
    operation_details: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (15, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize a complete tensor operation with inputs, process, and result.

    Creates a comprehensive visualization showing tensor operation workflow:
    input tensors → operation → result tensor. Includes index tracking
    and mathematical details.

    Args:
        operation_name: Name of the operation (e.g., "Contraction", "Symmetrization")
        input_tensors: Dictionary of input tensors
        result_tensor: Resulting tensor from the operation
        operation_details: Additional details about the operation
        figsize: Figure size for matplotlib plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    operation_details = operation_details or {}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f"Tensor Operation: {operation_name}", fontsize=16, fontweight="bold")

    # Input tensors (left column)
    ax_inputs = fig.add_subplot(gs[:, 0])
    ax_inputs.set_title("Input Tensors", fontweight="bold")

    y_pos = 0.9
    for name, tensor in input_tensors.items():
        # Tensor info
        info_text = f"{name}: Rank {tensor.rank}\n"
        info_text += f"Indices: {', '.join(tensor.indices.names)}\n"
        info_text += f"Types: {', '.join(tensor.indices.types)}\n"
        info_text += f"Shape: {tensor.shape}"

        ax_inputs.text(
            0.05,
            y_pos,
            info_text,
            transform=ax_inputs.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.3},
        )

        y_pos -= 0.4

    ax_inputs.axis("off")

    # Operation details (middle column, top)
    ax_operation = fig.add_subplot(gs[0, 1])
    ax_operation.set_title("Operation Details", fontweight="bold")

    operation_text = f"Operation: {operation_name}\n"

    if "contractions" in operation_details:
        operation_text += "Contractions:\n"
        for cont in operation_details["contractions"]:
            operation_text += f"  {cont}\n"

    if "symmetry" in operation_details:
        operation_text += f"Symmetry: {operation_details['symmetry']}\n"

    if "indices_traced" in operation_details:
        operation_text += f"Traced indices: {operation_details['indices_traced']}\n"

    ax_operation.text(
        0.05,
        0.95,
        operation_text,
        transform=ax_operation.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.3},
    )

    ax_operation.axis("off")

    # Mathematical formula (middle column, bottom)
    ax_formula = fig.add_subplot(gs[1, 1])
    ax_formula.set_title("Mathematical Expression", fontweight="bold")

    # Generate mathematical expression based on operation
    if operation_name.lower() == "contraction":
        formula = r"$C^{\alpha\beta} = A^{\alpha\mu} B_{\mu}^{\beta}$"
    elif operation_name.lower() == "trace":
        formula = r"$\text{Tr}(T) = T^{\mu}_{\mu} = \sum_{\mu} T^{\mu}_{\mu}$"
    elif operation_name.lower() == "symmetrization":
        formula = r"$T^{(\mu\nu)} = \frac{1}{2}(T^{\mu\nu} + T^{\nu\mu})$"
    elif operation_name.lower() == "antisymmetrization":
        formula = r"$T^{[\mu\nu]} = \frac{1}{2}(T^{\mu\nu} - T^{\nu\mu})$"
    else:
        formula = f"${operation_name}$"

    ax_formula.text(
        0.5, 0.5, formula, transform=ax_formula.transAxes, fontsize=14, ha="center", va="center"
    )

    ax_formula.axis("off")

    # Result tensor (right column)
    ax_result = fig.add_subplot(gs[:, 2])
    ax_result.set_title("Result Tensor", fontweight="bold")

    result_text = f"Result: Rank {result_tensor.rank}\n"
    result_text += f"Indices: {', '.join(result_tensor.indices.names)}\n"
    result_text += f"Types: {', '.join(result_tensor.indices.types)}\n"
    result_text += f"Shape: {result_tensor.shape}\n\n"

    # Show some component values if tensor is small enough
    if result_tensor.rank <= 2 and np.prod(result_tensor.shape) <= 16:
        result_text += "Components:\n"
        if result_tensor.rank == 0:
            result_text += f"{result_tensor.components:.6f}"
        else:
            # Flatten and show first few components
            components = result_tensor.components.flatten()
            for i, val in enumerate(components[:8]):  # Show max 8 components
                result_text += f"[{i}]: {val:.4f}\n"
            if len(components) > 8:
                result_text += "..."

    ax_result.text(
        0.05,
        0.95,
        result_text,
        transform=ax_result.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
    )

    ax_result.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_field_hierarchy(
    registry: AbstractFieldRegistry,
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the hierarchical relationships between fields in the theory.

    Creates a tree-like diagram showing physical fields, their response fields,
    and the relationships between them. Includes field properties, dimensions,
    and constraint information.

    Args:
        registry: Field registry containing all fields to visualize
        figsize: Figure size for matplotlib plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object

    Examples:
        >>> registry = create_registry_for_context("basic_physics")
        >>> fig = plot_field_hierarchy(registry)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create hierarchical graph
    G: nx.DiGraph = nx.DiGraph()

    # Add nodes for physical fields and their properties
    field_positions = {}
    field_names = registry.list_field_names()
    y_positions = np.linspace(0.8, 0.2, len(field_names))

    for i, name in enumerate(field_names):
        field = registry.get_field(name)
        if field is None:
            continue
        # Physical field position
        x_phys = 0.2
        y_pos = y_positions[i]
        field_positions[name] = (x_phys, y_pos)

        G.add_node(
            name,
            type="physical",
            rank=field.rank,
            dimension=field.dimension,
            canonical_dim=field.canonical_dimension,
            symmetric=field.properties.is_symmetric,
            traceless=field.properties.is_traceless,
            spatial=field.properties.is_spatial,
        )

        # Response field position
        response_name = f"tilde_{name}"
        x_resp = 0.8
        field_positions[response_name] = (x_resp, y_pos)

        G.add_node(
            response_name,
            type="response",
            rank=field.response.rank,
            dimension=field.response.dimension,
            canonical_dim=field.response.canonical_dimension,
            symmetric=field.response.properties.is_symmetric,
            traceless=field.response.properties.is_traceless,
            spatial=field.response.properties.is_spatial,
        )

        # Connect physical to response field
        G.add_edge(name, response_name, relation="response")

    # Draw nodes with different styles for physical vs response fields
    physical_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "physical"]
    response_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "response"]

    # Physical fields (blue boxes)
    for node in physical_nodes:
        x, y = field_positions[node]
        data = G.nodes[node]

        # Color by tensor rank
        rank = data["rank"]
        if rank == 0:
            color = "lightblue"
        elif rank == 1:
            color = "lightgreen"
        else:
            color = "lightcoral"

        # Draw field box
        box = patches.Rectangle(
            (x - 0.08, y - 0.05), 0.16, 0.1, facecolor=color, edgecolor="black", linewidth=2
        )
        ax.add_patch(box)

        # Field name and properties
        ax.text(x, y + 0.02, node, ha="center", va="center", fontsize=10, fontweight="bold")

        # Dimension info
        dim_text = f"[{data['dimension']:.1f}]"
        ax.text(x, y - 0.02, dim_text, ha="center", va="center", fontsize=8, style="italic")

        # Property indicators
        properties = []
        if data["symmetric"]:
            properties.append("S")
        if data["traceless"]:
            properties.append("T")
        if data["spatial"]:
            properties.append("⊥")

        if properties:
            prop_text = ",".join(properties)
            ax.text(x, y - 0.04, prop_text, ha="center", va="center", fontsize=7, color="green")

    # Response fields (orange boxes)
    for node in response_nodes:
        x, y = field_positions[node]
        data = G.nodes[node]

        # Draw response field box
        box = patches.Rectangle(
            (x - 0.08, y - 0.05),
            0.16,
            0.1,
            facecolor="orange",
            edgecolor="black",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(box)

        # Field name (use tilde notation)
        display_name = node.replace("tilde_", "~")
        ax.text(x, y + 0.02, display_name, ha="center", va="center", fontsize=10, fontweight="bold")

        # Dimension info
        dim_text = f"[{data['dimension']:.1f}]"
        ax.text(x, y - 0.02, dim_text, ha="center", va="center", fontsize=8, style="italic")

    # Draw arrows connecting physical to response fields
    for edge in G.edges(data=True):
        if edge[2]["relation"] == "response":
            x1, y1 = field_positions[edge[0]]
            x2, y2 = field_positions[edge[1]]

            arrow = patches.FancyArrowPatch(
                (x1 + 0.08, y1),
                (x2 - 0.08, y2),
                connectionstyle="arc3,rad=0",
                arrowstyle="->",
                mutation_scale=15,
                color="red",
                alpha=0.7,
                linewidth=2,
            )
            ax.add_patch(arrow)

    # Add field type labels
    ax.text(
        0.2,
        0.95,
        "Physical Fields",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )

    ax.text(
        0.8,
        0.95,
        "Response Fields (MSRJD)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "orange", "alpha": 0.7},
    )

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="lightblue", edgecolor="black", label="Scalar (rank 0)"),
        patches.Patch(facecolor="lightgreen", edgecolor="black", label="Vector (rank 1)"),
        patches.Patch(facecolor="lightcoral", edgecolor="black", label="Tensor (rank ≥ 2)"),
        patches.Patch(
            facecolor="orange", edgecolor="black", linestyle="--", label="Response Fields"
        ),
        plt.Line2D([0], [0], color="red", linewidth=2, label="Physical → Response"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.85))

    # Add property legend
    prop_text = "Property Codes:\nS = Symmetric\nT = Traceless\n⊥ = Spatial (orthogonal to u^μ)"
    ax.text(
        0.02,
        0.15,
        prop_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    # Add dimension explanation
    dim_text = "Dimensions:\n[n] = Engineering dimension\nResponse: [physical] - 4"
    ax.text(
        0.02,
        0.02,
        dim_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "lightcyan", "alpha": 0.8},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Israel-Stewart Field Hierarchy", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
