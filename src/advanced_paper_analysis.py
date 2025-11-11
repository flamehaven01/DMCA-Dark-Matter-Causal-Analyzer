# -*- coding: utf-8 -*-
"""
advanced_paper_analysis.py — Methodology/Uncertainty audit for DM-electron calculations.

Systematic uncertainty estimation for ab-initio dark matter scattering calculations.

This module implements the uncertainty checklist from:
    Dreyer et al., PRD 109 (2024) 095037
    "Systematic uncertainties in ab-initio calculations for direct DM-electron detection"

Key Uncertainty Sources:
1. **Basis set incompleteness**: Gaussian truncation errors
2. **XC functional**: DFT approximation (GGA vs. hybrid vs. meta-GGA)
3. **K-point sampling**: Brillouin zone discretization
4. **Real-space mesh**: FFT grid convergence
5. **Bloch sum cells**: Supercell size for high-q response
6. **High-q cutoff**: Breakdown of crystal form factor at |q| → ∞

Recommended Practices:
- Basis: gth-dzv minimum, gth-tzv2p for production, def2-TZVP for benchmarks
- XC: PBE baseline, PBEsol for better lattice constants, HSE06 for band gaps
- K-mesh: ≥(6,6,6) minimum, ≥(8,8,8) recommended for dense DOS near Fermi level
- Real-space: ≥(32,32,32) for GTH pseudopotentials
- Bloch cells: ≥4 for q > 0.5 a.u.⁻¹

Example:
    >>> run_meta = {
    ...     "basis": "gth-dzv",
    ...     "xc": "PBE",
    ...     "kmesh": (6, 6, 6),
    ...     "realspace_mesh": (36, 36, 36),
    ...     "bloch_cells": 4,
    ...     "uncertainty_budget": {
    ...         "basis_set": 0.01,
    ...         "xc_functional": 0.01,
    ...         "kmesh": 0.01,
    ...         "bloch_cells": 0.01,
    ...         "high_q_cut": 0.02
    ...     }
    ... }
    >>> auditor = UncertaintyAuditor.from_run(run_meta)
    >>> print(auditor.report())
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

# Recommended computational parameters (Dreyer et al. 2024)
RECOMMENDED = {
    "basis": ["gth-dzv", "gth-tzv2p", "def2-svp", "def2-tzvp"],
    "xc": ["PBE", "PBEsol", "HSE06"],
    "kmesh_min": (6, 6, 6),
    "mesh_min": (32, 32, 32),
    "bloch_cells_min": 4,
}


@dataclass
class UncertaintyAuditor:
    """
    Auditor for systematic uncertainties in DM-electron calculations.

    Attributes:
        run_meta: Dictionary of computational parameters
        checks: List of performed checks
        warnings: List of detected issues
        passes: List of validated parameters
    """
    run_meta: Dict[str, Any] = field(default_factory=dict)
    checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passes: List[str] = field(default_factory=list)

    @staticmethod
    def from_run(run_meta: Dict[str, Any]) -> "UncertaintyAuditor":
        """
        Create auditor from run metadata.

        Args:
            run_meta: Dictionary with keys like "basis", "xc", "kmesh", etc.

        Returns:
            UncertaintyAuditor: Initialized auditor instance

        Example:
            >>> meta = {"basis": "gth-dzv", "xc": "PBE", "kmesh": (8,8,8)}
            >>> auditor = UncertaintyAuditor.from_run(meta)
        """
        return UncertaintyAuditor(run_meta=run_meta)

    def _ge_tuple(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        """
        Check if tuple a >= tuple b element-wise.

        Args:
            a: First tuple
            b: Second tuple (must have same length)

        Returns:
            bool: True if all elements of a >= corresponding elements of b
        """
        if len(a) != len(b):
            return False
        return all(x >= y for x, y in zip(a, b))

    def audit(self) -> None:
        """
        Perform systematic uncertainty audit.

        Checks:
        1. Basis set adequacy
        2. XC functional choice
        3. K-point mesh density
        4. Real-space grid resolution
        5. Bloch supercell size
        6. Uncertainty budget completeness

        Modifies:
            self.checks: Appends performed checks
            self.warnings: Appends detected issues
            self.passes: Appends validated parameters
        """
        rm = self.run_meta

        # Extract parameters
        basis = rm.get("basis", "").lower()
        xc = rm.get("xc", "")
        kmesh = tuple(rm.get("kmesh", (0, 0, 0)))
        mesh = tuple(rm.get("realspace_mesh", (0, 0, 0)))
        bloch_cells = int(rm.get("bloch_cells", 1))

        # Log checks
        self.checks.append(f"Basis set: {basis}")
        self.checks.append(f"XC functional: {xc}")
        self.checks.append(f"K-point mesh: {kmesh}")
        self.checks.append(f"Real-space FFT grid: {mesh}")
        self.checks.append(f"Bloch supercell size: {bloch_cells}×{bloch_cells}×{bloch_cells}")

        # === 1. Basis Set ===
        if basis not in [b.lower() for b in RECOMMENDED["basis"]]:
            self.warnings.append(
                f"Basis '{basis}' not in recommended set {RECOMMENDED['basis']}. "
                f"Perform convergence test with larger basis (e.g., gth-tzv2p)."
            )
        else:
            self.passes.append(f"✓ Basis set '{basis}' in recommended set.")

        # === 2. XC Functional ===
        if xc not in RECOMMENDED["xc"]:
            self.warnings.append(
                f"XC functional '{xc}' not in recommended set {RECOMMENDED['xc']}. "
                f"For production, use PBE (baseline) or HSE06 (hybrid)."
            )
        else:
            self.passes.append(f"✓ XC functional '{xc}' validated.")

        # === 3. K-point Mesh ===
        if not self._ge_tuple(kmesh, RECOMMENDED["kmesh_min"]):
            self.warnings.append(
                f"K-mesh {kmesh} below recommended minimum {RECOMMENDED['kmesh_min']}. "
                f"Increase to ≥(8,8,8) for materials with dense DOS near Fermi level."
            )
        else:
            self.passes.append(f"✓ K-mesh {kmesh} meets minimum density.")

        # === 4. Real-space Grid ===
        if not self._ge_tuple(mesh, RECOMMENDED["mesh_min"]):
            self.warnings.append(
                f"Real-space grid {mesh} below recommended minimum {RECOMMENDED['mesh_min']}. "
                f"FFT integrals may be underconverged."
            )
        else:
            self.passes.append(f"✓ Real-space grid {mesh} adequate.")

        # === 5. Bloch Supercell ===
        if bloch_cells < RECOMMENDED["bloch_cells_min"]:
            self.warnings.append(
                f"Bloch supercell size {bloch_cells} below recommended minimum {RECOMMENDED['bloch_cells_min']}. "
                f"High-q response (|q| > 0.5 a.u.⁻¹) may exhibit finite-size artifacts. "
                f"Increase supercell or apply q-dependent corrections."
            )
        else:
            self.passes.append(f"✓ Bloch supercell size {bloch_cells} sufficient.")

        # === 6. Uncertainty Budget ===
        ub = rm.get("uncertainty_budget", {})
        required_keys = ["basis_set", "xc_functional", "kmesh", "bloch_cells", "high_q_cut"]

        missing = [k for k in required_keys if k not in ub]
        if missing:
            self.warnings.append(
                f"Missing uncertainty estimates for: {missing}. "
                f"Quantify systematic errors via convergence tests."
            )
        else:
            total_unc = sum(ub.values())
            self.passes.append(
                f"✓ Complete uncertainty budget provided (total: {total_unc:.2%})."
            )

    def report(self) -> str:
        """
        Generate human-readable audit report.

        Returns:
            str: Markdown-formatted report

        Example:
            >>> auditor.audit()
            >>> print(auditor.report())
            # DM–e⁻ Ab-Initio Run — Methodology Audit
            ...
        """
        # Run audit if not already done
        if not self.checks:
            self.audit()

        lines = []
        lines.append("# DM–e⁻ Ab-Initio Calculation — Methodology Audit\n")

        # Summary
        lines.append("## Summary")
        lines.append(f"- ✓ PASS: {len(self.passes)}")
        lines.append(f"- ⚠️ WARN: {len(self.warnings)}")
        lines.append(f"- 📋 CHECKS: {len(self.checks)}\n")

        # Checks performed
        lines.append("## Checks Performed\n")
        for check in self.checks:
            lines.append(f"- {check}")
        lines.append("")

        # Validated parameters
        if self.passes:
            lines.append("## ✓ Validated Parameters\n")
            for p in self.passes:
                lines.append(f"- {p}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## ⚠️ Warnings & Recommendations\n")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("**Reference**: Dreyer et al., PRD 109 (2024) 095037")
        lines.append("**Guideline**: All-electron, all-order ab-initio calculations")

        return "\n".join(lines)

    def quantify_total_uncertainty(self) -> Optional[float]:
        """
        Compute total systematic uncertainty (quadrature sum).

        Returns:
            float: Total relative uncertainty, or None if budget incomplete

        Example:
            >>> unc = auditor.quantify_total_uncertainty()
            >>> print(f"Total uncertainty: {unc:.2%}")
        """
        ub = self.run_meta.get("uncertainty_budget", {})
        if not ub:
            return None

        # Quadrature sum: σ_tot = √(Σ σ_i²)
        total = sum(v ** 2 for v in ub.values()) ** 0.5
        return float(total)
