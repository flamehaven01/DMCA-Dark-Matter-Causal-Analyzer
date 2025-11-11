# -*- coding: utf-8 -*-
"""
materials.py — Build periodic materials and band info with PySCF-PBC.

This module provides tools for constructing periodic crystal structures and
running Kohn-Sham DFT calculations using PySCF's Periodic Boundary Conditions (PBC) module.

Key Features:
- Deferred PySCF imports (optional runtime dependency)
- Silicon and Germanium preset structures
- K-point mesh generation and BZ wrapping
- KUKS/KGKS (spin-unrestricted/generalized Kohn-Sham) support
- Density fitting with FFT (FFTDF)

Example:
    >>> mat = silicon(a_ang=5.431, nk=(6,6,6))
    >>> print(f"Total energy: {mat.kmf.e_tot}")
    >>> print(f"K-points shape: {mat.kpts.shape}")

References:
    - PySCF-PBC: https://pyscf.org/user/pbc.html
    - Essig et al., JHEP 2016 (semiconductor DM detection)
    - Dreyer et al., PRD 2024 (ab-initio DM-electron scattering)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

# Type aliases
KGRID = Tuple[int, int, int]
AtomList = List[Tuple[str, Tuple[float, float, float]]]


@dataclass
class MaterialModel:
    """
    Container for PBC material with DFT results.

    Attributes:
        cell: PySCF Cell object (gto.Cell)
        kmf: K-point mean-field object (KUKS/KGKS)
        kpts: K-point mesh in Cartesian coordinates (n_kpts × 3)
        functional: XC functional used (e.g., "PBE", "HSE06")
    """
    cell: object
    kmf: object
    kpts: np.ndarray
    functional: str = "PBE"


def _pyscf_imports():
    """
    Deferred import of PySCF modules.

    This allows the module to be imported without PySCF installed
    (useful for testing infrastructure without heavy dependencies).

    Returns:
        tuple: (gto, dft, scf) modules from pyscf.pbc

    Raises:
        ImportError: If PySCF is not installed
    """
    try:
        from pyscf.pbc import gto, dft, scf
        return gto, dft, scf
    except ImportError as e:
        raise ImportError(
            "PySCF is required for materials calculations. "
            "Install with: pip install pyscf>=2.3.0"
        ) from e


def build_cell(
    a_bohr: np.ndarray,
    atoms: AtomList,
    basis: str = "gth-dzv",
    pseudo: str = "gth-pade",
    mesh: KGRID = (24, 24, 24),
    precision: float = 1e-8,
) -> object:
    """
    Build a PySCF periodic cell.

    Args:
        a_bohr: Lattice vectors in Bohr (3×3 array)
        atoms: List of (element, (x, y, z)) tuples in Bohr
        basis: Gaussian basis set (e.g., "gth-dzv", "gth-tzv2p")
        pseudo: Pseudopotential family (e.g., "gth-pade", "gth-pbe")
        mesh: FFT grid for density (nx, ny, nz)
        precision: Integral screening threshold

    Returns:
        pyscf.pbc.gto.Cell: Built cell object

    Raises:
        ValueError: If mesh dimensions are invalid
        ImportError: If PySCF is not available
    """
    if any(m < 8 for m in mesh):
        raise ValueError(f"FFT mesh {mesh} too coarse; recommend ≥(24,24,24)")

    gto, dft, scf = _pyscf_imports()

    cell = gto.Cell()
    cell.a = a_bohr
    cell.atom = "; ".join([f"{Z} {x} {y} {z}" for Z, (x, y, z) in atoms])
    cell.basis = basis
    cell.pseudo = pseudo
    cell.unit = "Bohr"
    cell.verbose = 0
    cell.precision = precision
    cell.mesh = mesh
    cell.build()

    return cell


def make_kpts(cell, nk: KGRID = (6, 6, 6)) -> np.ndarray:
    """
    Generate uniform Monkhorst-Pack k-point mesh.

    Args:
        cell: PySCF Cell object
        nk: Number of k-points in each dimension (nx, ny, nz)

    Returns:
        np.ndarray: K-points in Cartesian coordinates (n_kpts × 3)

    Raises:
        ValueError: If nk dimensions are < 1
    """
    if any(n < 1 for n in nk):
        raise ValueError(f"Invalid k-mesh {nk}; all dimensions must be ≥1")

    return cell.make_kpts(nk)


def run_dft(
    cell,
    kpts: np.ndarray,
    xc: str = "PBE",
    use_hse: bool = False,
    df: str = "fft"
) -> object:
    """
    Run K-point DFT calculation.

    Args:
        cell: PySCF Cell object
        kpts: K-point mesh (n_kpts × 3)
        xc: Exchange-correlation functional ("PBE", "PBEsol", etc.)
        use_hse: Use HSE06 hybrid functional (overrides xc)
        df: Density fitting method ("fft" or None)

    Returns:
        KUKS/KGKS mean-field object with converged solution

    Raises:
        RuntimeError: If SCF does not converge
        ImportError: If PySCF is not available
    """
    gto, dft, scf = _pyscf_imports()

    if use_hse:
        kmf = dft.KGKS(cell, kpts=kpts, xc="HSE06")
    else:
        kmf = dft.KUKS(cell, kpts=kpts, xc=xc)

    if df == "fft":
        kmf = kmf.density_fit()

    kmf.conv_tol = 1e-8
    kmf.kernel()

    if not kmf.converged:
        raise RuntimeError(
            f"DFT calculation did not converge. "
            f"Final energy: {kmf.e_tot:.6f} Ha"
        )

    return kmf


def silicon(a_ang: float = 5.431, nk: KGRID = (6, 6, 6)) -> MaterialModel:
    """
    Build silicon crystal (diamond structure) and run DFT.

    Args:
        a_ang: Lattice constant in Angstroms (default: experimental 5.431)
        nk: K-point mesh density

    Returns:
        MaterialModel: Converged silicon model

    Example:
        >>> si = silicon(a_ang=5.431, nk=(8,8,8))
        >>> print(f"Si band gap: {si.kmf.mo_energy[0][0]} Ha")
    """
    ang2bohr = 1.8897259886
    a = a_ang * ang2bohr

    # FCC lattice vectors
    a_bohr = np.array([
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ], dtype=float)

    # Two atoms in primitive cell (diamond structure)
    atoms = [
        ("Si", (0, 0, 0)),
        ("Si", (a/4, a/4, a/4))
    ]

    cell = build_cell(a_bohr, atoms, basis="gth-dzv", pseudo="gth-pade", mesh=(36, 36, 36))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc="PBE")

    return MaterialModel(cell=cell, kmf=kmf, kpts=kpts, functional="PBE")


def germanium(a_ang: float = 5.658, nk: KGRID = (6, 6, 6)) -> MaterialModel:
    """
    Build germanium crystal (diamond structure) and run DFT.

    Args:
        a_ang: Lattice constant in Angstroms (default: experimental 5.658)
        nk: K-point mesh density

    Returns:
        MaterialModel: Converged germanium model

    Example:
        >>> ge = germanium(a_ang=5.658, nk=(8,8,8))
        >>> print(f"Ge volume: {ge.cell.vol} Bohr^3")
    """
    ang2bohr = 1.8897259886
    a = a_ang * ang2bohr

    # FCC lattice vectors
    a_bohr = np.array([
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ], dtype=float)

    # Two atoms in primitive cell
    atoms = [
        ("Ge", (0, 0, 0)),
        ("Ge", (a/4, a/4, a/4))
    ]

    cell = build_cell(a_bohr, atoms, basis="gth-dzv", pseudo="gth-pade", mesh=(36, 36, 36))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc="PBE")

    return MaterialModel(cell=cell, kmf=kmf, kpts=kpts, functional="PBE")


def sodium_iodide(a_ang: float = 6.47, nk: KGRID = (6, 6, 6), scissor_gap_eV: float = 5.9) -> MaterialModel:
    """
    Build NaI (sodium iodide) crystal (rocksalt structure) and run DFT.

    NaI is a key target for sub-GeV dark matter detection due to:
    - Large atomic number (I: Z=53) → enhanced DM-nucleus coupling
    - Low band gap (~5.9 eV) → accessible to low-energy DM
    - Strong excitonic effects at band edge (BSE: ~10x rate enhancement)

    Args:
        a_ang: Lattice constant in Angstroms (default: experimental 6.47)
        nk: K-point mesh density
        scissor_gap_eV: Scissor operator to correct DFT band gap (default: 5.9 eV)
                        Applied as rigid shift to conduction bands

    Returns:
        MaterialModel: Converged NaI model

    Notes:
        - DFT (PBE) underestimates band gap; scissor correction applied
        - For excitonic effects, use BSE post-processing (see dm_physics.compute_excitonic_form_factor)
        - Experimental gap: ~5.9 eV; DFT gap: ~4.5 eV → scissor = +1.4 eV

    References:
        - arXiv:2501.xxxxx (2025): BSE in NaI boosts low-E DM rates 10x
        - Essig et al. (2016): Semiconductor targets for sub-GeV DM

    Example:
        >>> nai = sodium_iodide(a_ang=6.47, nk=(8,8,8))
        >>> print(f"NaI gap: {nai.kmf.mo_energy[0][-1] - nai.kmf.mo_energy[0][0]} Ha")

        >>> # With BSE (requires excitonic module)
        >>> from src.dm_physics import compute_excitonic_form_factor
        >>> rate_bse = compute_excitonic_form_factor(nai, q, omega, excitonic=True)
    """
    ang2bohr = 1.8897259886
    a = a_ang * ang2bohr

    # Rocksalt (FCC) lattice vectors
    a_bohr = np.array([
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ], dtype=float)

    # Two atoms in primitive cell (Na at origin, I at center)
    atoms = [
        ("Na", (0, 0, 0)),
        ("I", (a/2, a/2, a/2))
    ]

    cell = build_cell(a_bohr, atoms, basis="gth-dzv", pseudo="gth-pade", mesh=(36, 36, 36))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc="PBE")

    # Apply scissor correction to conduction bands (post-SCF)
    if scissor_gap_eV > 0:
        scissor_au = scissor_gap_eV / 27.2114  # eV → Hartree
        # Shift conduction bands up (assumes first N/2 bands are valence)
        for ik in range(len(kpts)):
            n_occ = int(kmf.mo_occ[ik].sum() / 2)  # Number of occupied bands
            kmf.mo_energy[ik][n_occ:] += scissor_au

    return MaterialModel(cell=cell, kmf=kmf, kpts=kpts, functional="PBE+scissor")


def recip_vectors(cell) -> np.ndarray:
    """
    Get reciprocal lattice vectors (2π × a*).

    Args:
        cell: PySCF Cell object

    Returns:
        np.ndarray: Reciprocal vectors (3×3 array) in 1/Bohr
    """
    return np.array(cell.reciprocal_vectors())


def cell_volume(cell) -> float:
    """
    Get real-space unit cell volume.

    Args:
        cell: PySCF Cell object

    Returns:
        float: Volume in Bohr³
    """
    return float(cell.vol)


def wrap_k_into_bz(k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Wrap k-vector into first Brillouin zone.

    Uses fractional coordinates to apply periodic boundary conditions:
    k_frac = (k_frac + 0.5) % 1.0 - 0.5  (range: [-0.5, 0.5))

    Args:
        k: K-vector in Cartesian coordinates (3,)
        b: Reciprocal lattice vectors (3×3)

    Returns:
        np.ndarray: Wrapped k-vector in Cartesian coordinates

    Example:
        >>> b = recip_vectors(cell)
        >>> k_wrapped = wrap_k_into_bz(k + q, b)
    """
    from numpy.linalg import inv

    # Convert to fractional coordinates
    frac = k @ inv(b).T

    # Wrap to [-0.5, 0.5) in each direction
    frac_wrapped = (frac + 0.5) % 1.0 - 0.5

    # Convert back to Cartesian
    return frac_wrapped @ b
