# -*- coding: utf-8 -*-
"""
visualization.py — Visualization utilities for DM scattering results.

Provides plotting functions for:
- dR/dω vs energy (scattering rate spectra)
- Form factors vs momentum transfer
- BSE enhancement comparisons
- Reach plots (cross-section vs mass)

Dependencies:
- matplotlib (already in requirements.txt)
- numpy
"""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple, Callable
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "Matplotlib not available. Install with: pip install matplotlib>=3.9",
        ImportWarning
    )


def _check_matplotlib():
    """Check if matplotlib is available and raise if not."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib>=3.9"
        )


def plot_scattering_rate(
    omega_array: np.ndarray,
    rate_array: np.ndarray,
    material_name: str = "Unknown",
    mchi_GeV: float = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot differential scattering rate dR/dω vs energy.

    Args:
        omega_array: Energy transfer array in eV
        rate_array: Scattering rate array (arbitrary units)
        material_name: Target material name (e.g., "NaI", "Si")
        mchi_GeV: DM mass in GeV (for title)
        ax: Existing axes to plot on (optional)
        show: Display plot interactively
        save_path: Save figure to file (e.g., "rate_nai.png")

    Returns:
        matplotlib.figure.Figure: Figure object

    Example:
        >>> import numpy as np
        >>> omega = np.linspace(1, 10, 100)  # eV
        >>> rate = np.exp(-(omega - 5)**2 / 2)  # Gaussian peak
        >>> fig = plot_scattering_rate(omega, rate, "NaI", mchi_GeV=10)
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(omega_array, rate_array, 'b-', linewidth=2, label=f'{material_name}')
    ax.set_xlabel('Energy Transfer $\\omega$ (eV)', fontsize=12)
    ax.set_ylabel('Rate $dR/d\\omega$ (arb. units)', fontsize=12)

    title = f'DM-Electron Scattering Rate: {material_name}'
    if mchi_GeV is not None:
        title += f' ($m_\\chi$ = {mchi_GeV} GeV)'
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Saved figure to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_bse_comparison(
    omega_array: np.ndarray,
    rate_dft: np.ndarray,
    rate_bse: np.ndarray,
    material_name: str = "NaI",
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare DFT vs BSE scattering rates (excitonic effects).

    Args:
        omega_array: Energy transfer array in eV
        rate_dft: DFT-only scattering rate
        rate_bse: BSE-corrected scattering rate
        material_name: Target material
        ax: Existing axes
        show: Display interactively
        save_path: Save path

    Returns:
        Figure object

    Example:
        >>> # Show ~10x enhancement at band edge
        >>> omega = np.linspace(5, 10, 100)
        >>> rate_dft = np.exp(-(omega - 6)**2 / 1)
        >>> rate_bse = 10 * rate_dft * (omega < 6.5) + rate_dft * (omega >= 6.5)
        >>> fig = plot_bse_comparison(omega, rate_dft, rate_bse, "NaI")
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(omega_array, rate_dft, 'b--', linewidth=2, label='DFT (no excitons)', alpha=0.7)
    ax.plot(omega_array, rate_bse, 'r-', linewidth=2, label='BSE (with excitons)')

    # Calculate enhancement factor
    with np.errstate(divide='ignore', invalid='ignore'):
        enhancement = rate_bse / rate_dft
        max_enhancement = np.nanmax(enhancement)
        if np.isfinite(max_enhancement):
            peak_idx = np.nanargmax(enhancement)
            peak_omega = omega_array[peak_idx]
            ax.axvline(peak_omega, color='gray', linestyle=':', alpha=0.5,
                      label=f'Peak enhancement: {max_enhancement:.1f}x @ {peak_omega:.1f} eV')

    ax.set_xlabel('Energy Transfer $\\omega$ (eV)', fontsize=12)
    ax.set_ylabel('Rate $dR/d\\omega$ (arb. units)', fontsize=12)
    ax.set_title(f'Excitonic Effects in {material_name}', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_form_factor(
    q_array: np.ndarray,
    form_factor_sq: np.ndarray,
    material_name: str = "Unknown",
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot crystal form factor |f(q)|² vs momentum transfer.

    Args:
        q_array: Momentum transfer |q| in 1/Bohr
        form_factor_sq: |f(q)|² (dimensionless)
        material_name: Target material
        ax: Existing axes
        show: Display interactively
        save_path: Save path

    Returns:
        Figure object

    Example:
        >>> q = np.linspace(0.01, 2, 100)  # 1/Bohr
        >>> F2 = np.exp(-q**2 / 0.5**2)  # Gaussian falloff
        >>> fig = plot_form_factor(q, F2, "Si")
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(q_array, form_factor_sq, 'g-', linewidth=2)
    ax.set_xlabel('Momentum Transfer $|q|$ (Bohr$^{-1}$)', fontsize=12)
    ax.set_ylabel('$|f(q)|^2$', fontsize=12)
    ax.set_title(f'Crystal Form Factor: {material_name}', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_reach(
    mchi_array: np.ndarray,
    sigma_limit: np.ndarray,
    experiments: List[Tuple[str, np.ndarray, np.ndarray]] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot DM reach: cross-section limit vs DM mass.

    Args:
        mchi_array: DM mass array in GeV
        sigma_limit: Cross-section limit in cm² (this work)
        experiments: List of (name, mass_array, limit_array) for comparison
        ax: Existing axes
        show: Display interactively
        save_path: Save path

    Returns:
        Figure object

    Example:
        >>> mchi = np.logspace(-1, 2, 100)  # 0.1 - 100 GeV
        >>> sigma = 1e-40 * (mchi / 1)**(-2)  # power law
        >>> fig = plot_reach(mchi, sigma, experiments=[
        ...     ("XENON1T", np.array([1, 10, 100]), np.array([1e-39, 1e-40, 1e-41]))
        ... ])
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # Plot this work
    ax.plot(mchi_array, sigma_limit, 'r-', linewidth=3, label='This Work', zorder=10)

    # Plot comparison experiments
    if experiments:
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (name, mass, limit) in enumerate(experiments):
            color = colors[i % len(colors)]
            ax.plot(mass, limit, '--', linewidth=2, label=name, color=color, alpha=0.7)

    ax.set_xlabel('DM Mass $m_\\chi$ (GeV)', fontsize=12)
    ax.set_ylabel('Cross Section $\\sigma_e$ (cm$^2$)', fontsize=12)
    ax.set_title('Dark Matter Reach', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_multi_material_comparison(
    omega_array: np.ndarray,
    rates_dict: dict,
    mchi_GeV: float = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare scattering rates across multiple materials.

    Args:
        omega_array: Energy transfer array in eV
        rates_dict: Dictionary {material_name: rate_array}
        mchi_GeV: DM mass in GeV
        show: Display interactively
        save_path: Save path

    Returns:
        Figure object

    Example:
        >>> omega = np.linspace(1, 10, 100)
        >>> rates = {
        ...     "NaI": 10 * np.exp(-(omega - 6)**2 / 1),
        ...     "Si": np.exp(-(omega - 4)**2 / 2),
        ...     "Ge": 2 * np.exp(-(omega - 3)**2 / 1.5)
        ... }
        >>> fig = plot_multi_material_comparison(omega, rates, mchi_GeV=10)
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (material, rate) in enumerate(rates_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(omega_array, rate, linewidth=2, label=material, color=color)

    ax.set_xlabel('Energy Transfer $\\omega$ (eV)', fontsize=12)
    ax.set_ylabel('Rate $dR/d\\omega$ (arb. units)', fontsize=12)

    title = 'Multi-Material Comparison'
    if mchi_GeV is not None:
        title += f' ($m_\\chi$ = {mchi_GeV} GeV)'
    ax.set_title(title, fontsize=14)

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig
