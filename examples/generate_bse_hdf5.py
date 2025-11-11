#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_bse_hdf5.py — Example generator for BSE external HDF5 files.

This script creates a template HDF5 file conforming to the DMCA BSE external
format specification (see docs/BSE_FORMAT.md). Users should adapt this script
to load their own BSE computation results (from qcmath, BerkeleyGW, etc.).

Usage:
    python examples/generate_bse_hdf5.py --material NaI --output data/nai_bse.h5

For production use:
    1. Replace placeholder form factors with real BSE computation results
    2. Update metadata (computation method, reference, date)
    3. Validate file with: python examples/validate_bse_file.py data/nai_bse.h5
"""
from __future__ import annotations
import argparse
import datetime
import numpy as np
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. Install with: pip install h5py", file=sys.stderr)
    sys.exit(1)


def generate_nai_bse(
    output_path: str,
    n_k: int = 64,
    n_q: int = 100,
    n_omega: int = 200,
    enhancement_factor: float = 10.0
) -> None:
    """
    Generate example NaI BSE HDF5 file.

    Args:
        output_path: Output file path (e.g., "data/nai_bse.h5")
        n_k: Number of k-points (default: 64 = 4×4×4 mesh)
        n_q: Number of q-points (default: 100)
        n_omega: Number of energy points (default: 200)
        enhancement_factor: Excitonic enhancement factor (default: 10x)

    Notes:
        This is a PLACEHOLDER. Replace form_factors with real BSE computation.
    """
    print(f"Generating NaI BSE template: {output_path}")
    print(f"  Grid: {n_k} k-points × {n_q} q-points × {n_omega} ω-points")

    # Material parameters (NaI rocksalt)
    a_ang = 6.47
    a_bohr = a_ang * 1.8897259886
    band_gap_eV = 5.9

    with h5py.File(output_path, "w") as f:
        # ============================================================
        # 1. K-POINTS (4×4×4 Monkhorst-Pack mesh)
        # ============================================================
        print("  [1/8] Generating k-points...")
        # Create uniform mesh in [0, 1)³ fractional coordinates
        nk1d = int(np.round(n_k ** (1/3)))
        k_frac = np.mgrid[0:nk1d, 0:nk1d, 0:nk1d].reshape(3, -1).T / nk1d
        k_frac = k_frac[:n_k]  # Truncate to exactly n_k points

        # Convert to Cartesian (units: 2π/a)
        b_mag = 2 * np.pi / a_bohr
        kpoints = k_frac * b_mag  # Shape: (n_k, 3)

        f.create_dataset("kpoints", data=kpoints, dtype="float64")
        print(f"    ✓ K-points: {kpoints.shape} (Cartesian, Bohr⁻¹)")

        # ============================================================
        # 2. Q-POINTS (momentum transfer |q| ∈ [0, 2] Bohr⁻¹)
        # ============================================================
        print("  [2/8] Generating q-points...")
        q_mag = np.linspace(0, 2.0, n_q)
        qpoints = np.column_stack([q_mag, np.zeros(n_q), np.zeros(n_q)])
        f.create_dataset("qpoints", data=qpoints, dtype="float64")
        print(f"    ✓ Q-points: {qpoints.shape} (along [100], Bohr⁻¹)")

        # ============================================================
        # 3. ENERGY TRANSFER (ω ∈ [0, 20] eV)
        # ============================================================
        print("  [3/8] Generating energy grid...")
        omega_eV = np.linspace(0, 20, n_omega)
        omega_Ha = omega_eV / 27.2114
        f.create_dataset("omega", data=omega_Ha, dtype="float64")
        print(f"    ✓ Energy: {omega_Ha.shape} (0-20 eV, Hartree)")

        # ============================================================
        # 4. DFT FORM FACTORS (baseline, no excitons)
        # ============================================================
        print("  [4/8] Generating DFT form factors (placeholder)...")
        # Placeholder: Gaussian centered at band gap
        omega_grid = omega_eV[:, None, None, None]  # Broadcast shape
        dft_ff = np.exp(-((omega_grid - band_gap_eV) / 2.0) ** 2)
        dft_ff = dft_ff.squeeze()  # Remove singleton dimensions

        # Reshape to (n_k, n_q, n_omega)
        dft_ff = np.broadcast_to(dft_ff, (n_k, n_q, n_omega)).copy()

        # Add q-dependence: decay as exp(-q²/2)
        q_mag_grid = np.linalg.norm(qpoints, axis=1)
        q_factor = np.exp(-q_mag_grid**2 / 2.0)
        dft_ff *= q_factor[None, :, None]

        f.create_dataset("dft_form_factors", data=dft_ff, dtype="float64", compression="gzip", compression_opts=4)
        print(f"    ✓ DFT form factors: {dft_ff.shape} (compressed)")

        # ============================================================
        # 5. BSE FORM FACTORS (with excitonic enhancement)
        # ============================================================
        print(f"  [5/8] Generating BSE form factors ({enhancement_factor}x enhancement)...")
        # Placeholder: BSE = DFT × enhancement near band gap
        omega_grid = omega_eV[:, None, None, None]
        # Enhancement only in excitonic window (5-10 eV for NaI)
        enhancement = np.ones_like(omega_grid)
        excitonic_mask = (omega_grid >= 5.0) & (omega_grid <= 10.0)
        enhancement[excitonic_mask] = enhancement_factor

        bse_ff = dft_ff * enhancement.squeeze()

        f.create_dataset("form_factors", data=bse_ff, dtype="float64", compression="gzip", compression_opts=4)
        print(f"    ✓ BSE form factors: {bse_ff.shape} (compressed)")
        print(f"      Peak enhancement: {(bse_ff / (dft_ff + 1e-12)).max():.1f}x")

        # ============================================================
        # 6. UNCERTAINTY ESTIMATES (10% baseline)
        # ============================================================
        print("  [6/8] Generating uncertainty estimates...")
        uncertainty = np.full_like(bse_ff, 0.10, dtype="float64")
        # Higher uncertainty at high q (supercell finite-size effects)
        q_factor_unc = 1 + 0.5 * (q_mag_grid / 2.0) ** 2
        uncertainty *= q_factor_unc[None, :, None]
        uncertainty = np.clip(uncertainty, 0.05, 0.50)  # 5-50%

        f.create_dataset("uncertainty", data=uncertainty, dtype="float64", compression="gzip", compression_opts=4)
        print(f"    ✓ Uncertainty: {uncertainty.shape} (5-50%, compressed)")

        # ============================================================
        # 7. RECIPROCAL LATTICE VECTORS
        # ============================================================
        print("  [7/8] Adding reciprocal lattice vectors...")
        b = 2 * np.pi / a_bohr * np.eye(3)  # Simple cubic reciprocal lattice
        f.create_dataset("reciprocal_lattice_vectors", data=b, dtype="float64")
        print(f"    ✓ Reciprocal lattice: {b.shape} (Bohr⁻¹)")

        # ============================================================
        # 8. METADATA
        # ============================================================
        print("  [8/8] Adding metadata...")
        meta = f.create_group("metadata")
        meta.attrs["material_name"] = "NaI"
        meta.attrs["lattice_constant_bohr"] = a_bohr
        meta.attrs["structure_type"] = "rocksalt"
        meta.attrs["num_atoms"] = 2
        meta.attrs["band_gap_eV"] = band_gap_eV
        meta.attrs["computation_method"] = "PLACEHOLDER (user must replace with real BSE computation)"
        meta.attrs["date_computed"] = datetime.datetime.now().isoformat()
        meta.attrs["reference"] = "This is a TEMPLATE. Replace with actual BSE computation reference."
        meta.attrs["generator"] = "DMCA examples/generate_bse_hdf5.py"
        meta.attrs["version"] = "1.0"
        print("    ✓ Metadata added")

    print(f"\n✓ Generated: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024**2:.1f} MB")
    print("\n⚠️  WARNING: This file contains PLACEHOLDER form factors!")
    print("   Replace with real BSE computation before production use.")
    print("   See docs/BSE_FORMAT.md for details.\n")


def generate_csi_bse(output_path: str) -> None:
    """Generate example CsI BSE HDF5 file (moderate excitonic effects)."""
    print("Generating CsI BSE template (3-5x enhancement)...")
    # CsI has weaker excitonic effects than NaI
    generate_nai_bse(output_path, enhancement_factor=4.0)
    print("✓ CsI template generated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate example BSE HDF5 files for DMCA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate NaI BSE file (10x enhancement)
  python examples/generate_bse_hdf5.py --material NaI --output data/nai_bse.h5

  # Generate CsI BSE file (4x enhancement)
  python examples/generate_bse_hdf5.py --material CsI --output data/csi_bse.h5

  # High-density grid for production
  python examples/generate_bse_hdf5.py --material NaI --output data/nai_bse_dense.h5 \\
      --n-k 216 --n-q 200 --n-omega 500

Notes:
  - Generated files contain PLACEHOLDER data. Replace with real BSE computation.
  - See docs/BSE_FORMAT.md for full specification.
  - Validate with: python examples/validate_bse_file.py <file.h5>
        """
    )
    parser.add_argument("--material", type=str, default="NaI",
                        choices=["NaI", "CsI"],
                        help="Material to generate (default: NaI)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--n-k", type=int, default=64,
                        help="Number of k-points (default: 64)")
    parser.add_argument("--n-q", type=int, default=100,
                        help="Number of q-points (default: 100)")
    parser.add_argument("--n-omega", type=int, default=200,
                        help="Number of energy points (default: 200)")

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate file
    if args.material == "NaI":
        generate_nai_bse(str(output_path), n_k=args.n_k, n_q=args.n_q, n_omega=args.n_omega)
    elif args.material == "CsI":
        generate_csi_bse(str(output_path))
    else:
        print(f"ERROR: Unknown material '{args.material}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
