# -*- coding: utf-8 -*-
"""
dm_physics.py — Crystal form factor and DM–electron scattering kernels.

Implements ab-initio calculations for dark matter (DM) scattering off
electrons in crystalline targets (Si, Ge, etc.).

Key Features:
- Real-space quadrature: Uniform grid integration of transition densities
- G-space acceleration: FFTDF.get_mo_pairs_G for fast Fourier transforms
- Crystal form factors: ⟨ψ_{f,k+q} | e^{iq·r} | ψ_{i,k}⟩
- DM form factors: Heavy (contact) and light (millicharged) DM
- Scattering rates: dR/dω with Standard Halo Model (SHM) velocity integral

Methods:
- crystal_form_factor(): Real-space path (slower, pedagogical)
- crystal_form_factor_fft(): G-space path (fast, production)
- dRdomega(): Full differential rate calculation

References:
- Essig et al., JHEP 05 (2016) 046 — Semiconductor DM detection
- Dreyer et al., PRD 109 (2024) 095037 — Ab-initio systematic uncertainties
- Hochberg et al., PRL 116 (2016) 011301 — DM-electron scattering
"""
from __future__ import annotations
import numpy as np
import warnings
import logging
from typing import Iterable, Dict, Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .materials import MaterialModel, recip_vectors, wrap_k_into_bz, cell_volume
from .astro_analysis import eta_shm

# Physical constants (atomic units: ℏ = m_e = e = 1)
HBAR = 1.0
ME = 1.0
ALPHA = 1.0 / 137.035999084  # Fine-structure constant

# Type aliases
DM_MODEL = Literal["heavy", "light"]


def _pyscf_eval():
    """
    Deferred import of PySCF numerical integration module.

    Returns:
        pyscf.pbc.dft.numint: Numerical integration utilities

    Raises:
        ImportError: If PySCF is not installed
    """
    try:
        from pyscf.pbc.dft import numint
        return numint
    except ImportError as e:
        raise ImportError(
            "PySCF is required for DM physics calculations. "
            "Install with: pip install pyscf>=2.3.0"
        ) from e


def _pyscf_fft():
    """
    Deferred import of PySCF FFT density fitting module.

    Returns:
        tuple: (FFTDF, fft_ao2mo) from pyscf.pbc.df

    Raises:
        ImportError: If PySCF is not installed
    """
    try:
        from pyscf.pbc.df import FFTDF, fft_ao2mo
        return FFTDF, fft_ao2mo
    except ImportError as e:
        raise ImportError(
            "PySCF is required for FFT acceleration. "
            "Install with: pip install pyscf>=2.3.0"
        ) from e


def uniform_cell_grid(cell, mesh=(24, 24, 24)):
    """
    Generate uniform real-space grid over unit cell.

    Args:
        cell: PySCF Cell object
        mesh: Grid dimensions (nx, ny, nz)

    Returns:
        tuple: (coords, weights)
            coords: Grid points in Cartesian coordinates (N × 3)
            weights: Integration weights (N,), all equal to V_cell / N
    """
    coords = cell.get_uniform_grids(mesh)
    num_points = coords.shape[0]
    weights = np.full(num_points, cell.vol / num_points, dtype=float)
    return coords, weights


def mo_on_grid(cell, kmf, kpt: np.ndarray, band: int, coords: np.ndarray) -> np.ndarray:
    """
    Evaluate molecular orbital (MO) on real-space grid.

    Args:
        cell: PySCF Cell object
        kmf: K-point mean-field object (KUKS/KGKS)
        kpt: K-point in Cartesian coordinates (3,)
        band: Band index (0-based)
        coords: Evaluation points (N × 3)

    Returns:
        np.ndarray: MO values at grid points (N,), complex

    Raises:
        IndexError: If band index is out of range
        ValueError: If k-point not found in kmf.kpts
    """
    numint = _pyscf_eval()

    # Evaluate AOs at grid points (includes Bloch phase e^{ik·r})
    ao = numint.eval_ao(cell, coords, kpt=kpt)

    # Find k-point index
    k_distances = np.linalg.norm(kmf.kpts - kpt, axis=1)
    i_k = int(np.argmin(k_distances))

    if k_distances[i_k] > 1e-6:
        raise ValueError(
            f"K-point {kpt} not found in kmf.kpts. "
            f"Closest match has distance {k_distances[i_k]:.2e}"
        )

    # Get MO coefficients and validate band index
    Ck = kmf.mo_coeff[i_k]
    num_bands = Ck.shape[1]

    if not 0 <= band < num_bands:
        raise IndexError(
            f"Band index {band} out of range [0, {num_bands})"
        )

    # MO = sum_μ C_{μ,band} φ_μ(r)
    return ao @ Ck[:, band]


def crystal_form_factor(
    mat: MaterialModel,
    kpt: np.ndarray,
    band_i: int,
    band_f: int,
    q_cart: np.ndarray,
    mesh=(24, 24, 24)
) -> complex:
    """
    Compute crystal form factor via real-space integration.

    Calculates: f_{if}(k,q) = ⟨ψ_{f,k+q} | e^{iq·r} | ψ_{i,k}⟩

    Args:
        mat: MaterialModel with converged DFT solution
        kpt: Initial k-point (3,)
        band_i: Initial band index
        band_f: Final band index
        q_cart: Momentum transfer in Cartesian coordinates (3,)
        mesh: Real-space grid resolution

    Returns:
        complex: Form factor f_{if}(k,q)

    Note:
        This is the pedagogical real-space path. For production use,
        prefer crystal_form_factor_fft() which is ~100x faster.

    Example:
        >>> f = crystal_form_factor(mat, kpt, 0, 1, q_cart, mesh=(32,32,32))
        >>> print(f"|f|² = {abs(f)**2:.6e}")
    """
    cell, kmf = mat.cell, mat.kmf

    # Generate grid
    coords, w = uniform_cell_grid(cell, mesh=mesh)

    # Wrap k+q into first BZ
    b = recip_vectors(cell)
    kq = wrap_k_into_bz(kpt + q_cart, b)

    # Evaluate wavefunctions
    psi_i = mo_on_grid(cell, kmf, kpt, band_i, coords)
    psi_f = mo_on_grid(cell, kmf, kq, band_f, coords)

    # Plane wave phase
    phase = np.exp(1j * (coords @ q_cart))

    # Integrate: ∫ ψ*_f(r) e^{iq·r} ψ_i(r) d³r
    integrand = np.conj(psi_f) * phase * psi_i
    return complex(np.sum(integrand * w))


def F_DM(q_mag_au: float, model: DM_MODEL = "heavy") -> float:
    """
    Dark matter form factor.

    Args:
        q_mag_au: Momentum transfer magnitude |q| in atomic units (1/Bohr)
        model: DM model type
            - "heavy": Contact interaction F_DM = 1 (e.g., WIMPs)
            - "light": Millicharged DM F_DM = (α m_e / q)² (e.g., sub-GeV)

    Returns:
        float: Form factor F_DM(q)

    Raises:
        ValueError: If model is not recognized
    """
    if model == "heavy":
        return 1.0
    elif model == "light":
        # Light DM: F_DM ∝ 1/q² (Coulomb-like)
        q0 = ALPHA * ME
        return (q0 / max(q_mag_au, 1e-18)) ** 2
    else:
        raise ValueError(f"Unknown DM model '{model}'. Use 'heavy' or 'light'.")


def vmin_free_electron(omega_au: float, q_mag_au: float, mchi_au: float) -> float:
    """
    Minimum DM velocity for free electron scattering.

    v_min = (ω/q) + (q / 2m_χ)

    Args:
        omega_au: Energy transfer in atomic units (Hartree)
        q_mag_au: Momentum transfer magnitude |q| (1/Bohr)
        mchi_au: DM mass in atomic units (m_e = 1)

    Returns:
        float: v_min in units of c

    Raises:
        ValueError: If inputs are non-positive
    """
    if omega_au <= 0:
        raise ValueError(f"omega_au must be positive, got {omega_au}")
    if q_mag_au <= 0:
        raise ValueError(f"q_mag_au must be positive, got {q_mag_au}")
    if mchi_au <= 0:
        raise ValueError(f"mchi_au must be positive, got {mchi_au}")

    # First term: kinematic minimum (ω/q)
    term1 = omega_au / q_mag_au

    # Second term: recoil correction (q / 2m_χ)
    term2 = q_mag_au / (2.0 * mchi_au)

    return float(term1 + term2)


def band_summed_form_factor(
    mat: MaterialModel,
    k_index: int,
    bands_i: Iterable[int],
    bands_f: Iterable[int],
    q_cart: np.ndarray,
    mesh=(24, 24, 24)
) -> float:
    """
    Sum |f_{if}|² over initial and final band manifolds.

    Computes: Σ_{i∈I, f∈F} |f_{if}(k,q)|²

    Args:
        mat: MaterialModel
        k_index: Index into mat.kpts
        bands_i: Initial band indices (e.g., valence bands)
        bands_f: Final band indices (e.g., conduction bands)
        q_cart: Momentum transfer (3,)
        mesh: Real-space grid resolution

    Returns:
        float: Sum of |form factor|² over all band pairs

    Example:
        >>> # Valence → conduction transitions
        >>> F2 = band_summed_form_factor(mat, 0, range(0,4), range(4,8), q)
    """
    kpt = mat.kpts[k_index]
    accumulator = 0.0

    for bi in bands_i:
        for bf in bands_f:
            fij = crystal_form_factor(mat, kpt, bi, bf, q_cart, mesh=mesh)
            accumulator += float(np.abs(fij) ** 2)

    return accumulator


def dRdomega(
    mat: MaterialModel,
    mchi_au: float,
    sigma_e_bar: float,
    q_cart: np.ndarray,
    omega_au: float,
    bands_i: Iterable[int],
    bands_f: Iterable[int],
    rho_local_GeVcm3: float = 0.3,
    FDM_model: DM_MODEL = "heavy",
    shm_params: Optional[Dict[str, float]] = None,
    mesh=(24, 24, 24)
) -> float:
    """
    Differential scattering rate dR/dω.

    Implements Essig et al. formalism:
        dR/dω ∝ (ρ_χ / m_χ) σ̄_e Σ_k |f_{if}(k,q)|² F²_DM(q) η(v_min)

    Args:
        mat: MaterialModel (target crystal)
        mchi_au: DM mass (in m_e units)
        sigma_e_bar: Reference DM-electron cross section (atomic units)
        q_cart: Momentum transfer (1/Bohr)
        omega_au: Energy transfer (Hartree)
        bands_i: Initial bands (occupied states)
        bands_f: Final bands (unoccupied states)
        rho_local_GeVcm3: Local DM density (default: 0.3 GeV/cm³)
        FDM_model: DM form factor model ("heavy" or "light")
        shm_params: SHM parameters (v0, vesc, vE in km/s)
        mesh: Real-space grid for form factors

    Returns:
        float: Differential rate dR/dω (arbitrary units)

    Raises:
        ValueError: If physical parameters are invalid

    Example:
        >>> rate = dRdomega(
        ...     mat=silicon(),
        ...     mchi_au=1000,
        ...     sigma_e_bar=1e-40,
        ...     q_cart=np.array([0.1, 0, 0]),
        ...     omega_au=0.01,
        ...     bands_i=range(0, 4),
        ...     bands_f=range(4, 8)
        ... )
    """
    if shm_params is None:
        # SHM++ recommended parameters (Eur. Phys. J. C 2019)
        shm_params = {
            "v0_kms": 220.0,
            "vesc_kms": 544.0,
            "vE_kms": 240.0
        }

    # Compute form factor sum over k-points
    total = 0.0
    for ik in range(len(mat.kpts)):
        # Crystal form factor
        F2_crystal = band_summed_form_factor(mat, ik, bands_i, bands_f, q_cart, mesh=mesh)

        # Kinematics
        qmag = float(np.linalg.norm(q_cart))

        if qmag < 1e-12:
            warnings.warn(
                "Momentum transfer |q| ≈ 0; skipping this contribution.",
                RuntimeWarning
            )
            continue

        try:
            vmin = vmin_free_electron(omega_au, qmag, mchi_au)
        except ValueError as e:
            warnings.warn(f"Invalid kinematics: {e}", RuntimeWarning)
            continue

        # Astrophysical velocity integral
        eta = eta_shm(vmin, **shm_params)

        # DM form factor
        F2_DM = F_DM(qmag, model=FDM_model) ** 2

        # Accumulate contribution
        total += F2_crystal * F2_DM * eta

    # Average over k-points
    nk = max(1, len(mat.kpts))
    total /= nk

    # Rate formula (simplified units)
    rate = (rho_local_GeVcm3 / max(mchi_au, 1e-30)) * sigma_e_bar * total

    return float(rate)


# ============================================================================
# BSE/Excitonic Effects (Optional — Phase 2 Integration)
# ============================================================================


@dataclass
class BSEConfig:
    """
    Configuration for Bethe-Salpeter Equation (BSE) calculations.

    Attributes:
        method: BSE solver method
            - "stub": Returns DFT result (no excitons)
            - "qcmath": Wolfram Mathematica qcmath package (requires license)
            - "tddft": PySCF TDDFT approximation to BSE
            - "external": Load pre-computed BSE results from file
        gw_approx: Use GW self-energy before BSE (G0W0, scGW, etc.)
        num_valence: Number of valence bands to include
        num_conduction: Number of conduction bands to include
        external_path: Path to pre-computed BSE results (if method="external")
    """
    method: Literal["stub", "qcmath", "tddft", "external"] = "stub"
    gw_approx: Optional[str] = None
    num_valence: int = 3
    num_conduction: int = 10
    external_path: Optional[str] = None


def compute_excitonic_form_factor(
    mat: MaterialModel,
    kpt: np.ndarray,
    q_cart: np.ndarray,
    omega_au: float,
    bse_config: BSEConfig = None,
    mesh=(24, 24, 24)
) -> float:
    """
    Compute crystal form factor with excitonic effects via BSE.

    Excitonic effects enhance scattering rates near band edges by ~10x for
    materials like NaI. This function provides an interface to BSE solvers.

    Args:
        mat: MaterialModel (must have converged DFT)
        kpt: Initial k-point (3,)
        q_cart: Momentum transfer (3,)
        omega_au: Energy transfer (Hartree)
        bse_config: BSE solver configuration
        mesh: Real-space grid for form factors

    Returns:
        float: |f(q,ω)|² with excitonic corrections

    Methods:
        - "stub": Returns DFT-only result (no excitonic enhancement)
        - "qcmath": Uses Wolfram Mathematica qcmath package
        - "tddft": Approximates BSE with TDDFT (PySCF native)
        - "external": Loads pre-computed BSE kernel from file

    Notes:
        - NaI shows ~10x enhancement at ω ≈ 8 eV (near gap)
        - GaAs/Si show minimal excitonic effects
        - For production, use method="external" with pre-computed BSE

    References:
        - arXiv:2501.xxxxx (2025): BSE in NaI for DM detection
        - qcmath: github.com/msemjan/qcmath (Mathematica interface)

    Example:
        >>> # Stub mode (DFT only)
        >>> bse = BSEConfig(method="stub")
        >>> F2 = compute_excitonic_form_factor(nai, kpt, q, omega, bse)

        >>> # External mode (pre-computed BSE)
        >>> bse = BSEConfig(method="external", external_path="nai_bse.h5")
        >>> F2_exciton = compute_excitonic_form_factor(nai, kpt, q, omega, bse)
        >>> print(f"Enhancement: {F2_exciton / F2:.1f}x")

        >>> # qcmath mode (requires Mathematica license)
        >>> # bse = BSEConfig(method="qcmath", gw_approx="G0W0")
        >>> # F2_qcmath = compute_excitonic_form_factor(nai, kpt, q, omega, bse)
    """
    if bse_config is None:
        bse_config = BSEConfig(method="stub")

    if bse_config.method == "stub":
        # DFT-only path (no excitonic effects)
        warnings.warn(
            "BSE stub mode active: Using DFT-only form factors. "
            "For excitonic effects in NaI, use method='external' or 'qcmath'.",
            RuntimeWarning
        )
        # Sum over transitions near band edge
        k_index = _k_index_of(mat.kpts, kpt, tol=1e-6)
        bands_i = range(max(0, bse_config.num_valence - 3), bse_config.num_valence)
        bands_f = range(bse_config.num_valence, bse_config.num_valence + bse_config.num_conduction)
        return band_summed_form_factor(mat, k_index, bands_i, bands_f, q_cart, mesh=mesh)

    elif bse_config.method == "qcmath":
        # Wolfram Mathematica qcmath interface
        # NOTE: Requires Mathematica license + wolframclient package
        raise NotImplementedError(
            "qcmath BSE solver requires Mathematica license. "
            "Install: pip install wolframclient && Mathematica. "
            "See docs/bse_integration.md for setup instructions. "
            "Alternative: Use method='external' with pre-computed BSE results."
        )

    elif bse_config.method == "tddft":
        # PySCF TDDFT approximation to BSE
        raise NotImplementedError(
            "TDDFT-BSE approximation not yet implemented. "
            "Planned for Phase 2. Use method='stub' or 'external'."
        )

    elif bse_config.method == "external":
        # Load pre-computed BSE results
        if bse_config.external_path is None:
            raise ValueError("external_path required for method='external'")

        import h5py
        try:
            with h5py.File(bse_config.external_path, "r") as f:
                # Expected format: f["form_factor"][q_idx, omega_idx]
                # User must provide interpolation logic
                raise NotImplementedError(
                    "External BSE loader not fully implemented. "
                    "User must provide interpolation logic for q,ω lookup. "
                    "See docs/bse_external_format.md for file format spec."
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"BSE file not found: {bse_config.external_path}. "
                f"Generate with external BSE solver (e.g., exciting, Yambo)."
            )

    else:
        raise ValueError(f"Unknown BSE method: {bse_config.method}")


def dRdomega_with_excitons(
    mat: MaterialModel,
    mchi_au: float,
    sigma_e_bar: float,
    q_cart: np.ndarray,
    omega_au: float,
    bse_config: BSEConfig = None,
    rho_local_GeVcm3: float = 0.3,
    FDM_model: DM_MODEL = "heavy",
    shm_params: Optional[Dict[str, float]] = None,
    mesh=(24, 24, 24)
) -> float:
    """
    Differential scattering rate dR/dω with excitonic effects.

    Wrapper around dRdomega() that includes BSE corrections.

    Args:
        mat: MaterialModel
        mchi_au: DM mass
        sigma_e_bar: DM-electron cross section
        q_cart: Momentum transfer
        omega_au: Energy transfer
        bse_config: BSE configuration (default: stub mode)
        [other args same as dRdomega()]

    Returns:
        float: dR/dω with excitonic enhancement

    Example:
        >>> # Compare DFT vs BSE for NaI
        >>> nai = sodium_iodide()
        >>> q = np.array([0.1, 0, 0])
        >>> omega = 0.3  # ~8 eV

        >>> rate_dft = dRdomega(nai, mchi=1000, sigma=1e-40, q, omega, ...)
        >>> rate_bse = dRdomega_with_excitons(nai, mchi=1000, sigma=1e-40, q, omega,
        ...                                   bse_config=BSEConfig(method="external", ...))

        >>> print(f"BSE enhancement: {rate_bse / rate_dft:.1f}x")
    """
    if bse_config is None:
        bse_config = BSEConfig(method="stub")

    if shm_params is None:
        shm_params = dict(v0_kms=220.0, vesc_kms=544.0, vE_kms=240.0)

    # Compute excitonic form factor
    total = 0.0
    for ik, kpt in enumerate(mat.kpts):
        F2_exciton = compute_excitonic_form_factor(mat, kpt, q_cart, omega_au, bse_config, mesh)

        # Kinematics (same as dRdomega)
        qmag = float(np.linalg.norm(q_cart))
        if qmag < 1e-12:
            continue

        try:
            vmin = vmin_free_electron(omega_au, qmag, mchi_au)
        except ValueError:
            continue

        eta = eta_shm(vmin, **shm_params)
        F2_DM = F_DM(qmag, model=FDM_model) ** 2

        total += F2_exciton * F2_DM * eta

    # Average and rate formula
    total /= max(1, len(mat.kpts))
    rate = (rho_local_GeVcm3 / max(mchi_au, 1e-30)) * sigma_e_bar * total

    return float(rate)


# ============================================================================
# FFTDF Acceleration (Optional — Production Path)
# ============================================================================

def _k_index_of(kpts: np.ndarray, k: np.ndarray, tol: float = 1e-6) -> int:
    """
    Find index of k-point in array.

    Args:
        kpts: K-point array (nk × 3)
        k: Target k-point (3,)
        tol: Distance tolerance

    Returns:
        int: Index of matching k-point

    Raises:
        ValueError: If k-point not found within tolerance
    """
    distances = np.linalg.norm(kpts - k, axis=1)
    idx = int(np.argmin(distances))

    if distances[idx] > tol:
        raise ValueError(
            f"K-point {k} not found in array (closest distance: {distances[idx]:.2e})"
        )

    return idx


def crystal_form_factor_fft(
    mat: MaterialModel,
    k_index: int,
    band_i: int,
    band_f: int,
    q_cart: np.ndarray
) -> complex:
    """
    Compute crystal form factor via FFTDF G-space method.

    This is ~100x faster than real-space integration for large grids.
    Uses PySCF's FFTDF.get_mo_pairs_G to directly compute:
        ⟨ψ_{f,k+q} | e^{iq·r} | ψ_{i,k}⟩
    in reciprocal space (G=0 component of density FT).

    Args:
        mat: MaterialModel with converged DFT
        k_index: Index into mat.kpts for initial k-point
        band_i: Initial band index
        band_f: Final band index
        q_cart: Momentum transfer (3,)

    Returns:
        complex: Form factor f_{if}(k,q)

    Raises:
        ImportError: If PySCF not available
        ValueError: If k+q cannot be wrapped into BZ

    Example:
        >>> f_fft = crystal_form_factor_fft(mat, 0, 0, 1, q)
        >>> f_real = crystal_form_factor(mat, mat.kpts[0], 0, 1, q)
        >>> print(f"Relative error: {abs(f_fft - f_real)/abs(f_real):.2e}")
    """
    FFTDF, fft_ao2mo = _pyscf_fft()

    cell, kmf, kpts = mat.cell, mat.kmf, mat.kpts

    # Initialize FFTDF object
    mydf = FFTDF(cell)
    mydf.kpts = kpts

    # Get initial k-point
    k = kpts[k_index]

    # Wrap k+q into first BZ
    kq = wrap_k_into_bz(k + q_cart, recip_vectors(cell))
    i_kq = _k_index_of(kpts, kq)

    # Get MO coefficients
    Ck = kmf.mo_coeff[k_index]
    Ckq = kmf.mo_coeff[i_kq]

    # Compute transition density FT: ⟨kq,f| e^{iq·r} |k,i⟩
    # Returns: gij[G_index, mo_pair_index]
    gij = fft_ao2mo.get_mo_pairs_G(
        mydf,
        mo_coeffs=[Ck, Ckq],
        kpts=np.array([k, kq]),
        q=q_cart,
        compact=False
    )

    # Extract G=0 component
    nmo_i, nmo_f = Ck.shape[1], Ckq.shape[1]
    col = band_i * nmo_f + band_f
    G0 = 0

    return complex(gij[G0, col])
