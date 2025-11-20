# -*- coding: utf-8 -*-
"""
src/materials/base.py — Core classes and utilities for PySCF materials.

Part of the DFI-Meta Structural Evolution (Cycle 1).
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
