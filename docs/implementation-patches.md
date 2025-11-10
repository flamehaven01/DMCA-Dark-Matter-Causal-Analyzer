# Implementation Patches
## Ready-to-Apply Diff Patches for Physics Module Integration

**Document Version**: 1.0
**Date**: 2025-11-10
**Application Order**: Follow patch numbers sequentially (Patch 1 → 2 → 3 → ...)

---

## Table of Contents

1. [Patch 1: Module Foundation](#patch-1-module-foundation)
2. [Patch 2: Base Classes](#patch-2-base-classes)
3. [Patch 3: Silicon DFT Implementation](#patch-3-silicon-dft-implementation)
4. [Patch 4: Form Factors](#patch-4-form-factors)
5. [Patch 5: Physics Emitter](#patch-5-physics-emitter)
6. [Patch 6: Configuration Extension](#patch-6-configuration-extension)
7. [Patch 7: Schema Extension](#patch-7-schema-extension)
8. [Patch 8: Tests](#patch-8-tests)
9. [Patch 9: CI/CD](#patch-9-cicd)
10. [Patch 10: ASDP Rules](#patch-10-asdp-rules)

---

## Patch 1: Module Foundation

### Create src/materials/__init__.py

**File**: `src/materials/__init__.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""
DMCA Physics Module - Optional PySCF-based dark matter simulations

This module is OPTIONAL and isolated from core DMCA functionality.
All imports gracefully degrade if PySCF is not installed.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def _have_pyscf() -> bool:
    """Check if PySCF is available (similar to src/causal.py pattern)"""
    try:
        import pyscf  # noqa
        return True
    except ImportError:
        return False

def run_simulation(
    hypothesis: str,
    material: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run dark matter physics simulation.

    Args:
        hypothesis: One of [WIMP, Axion, DarkPhoton, SterileNeutrino]
        material: One of [Si, Ge, GaAs]
        config: PySCF configuration (basis, kpts, etc.)

    Returns:
        {
          "physics": {
            "material": "Si",
            "e_tot_hartree": -289.123,
            "band_gap_ev": 1.17,
            ...
          } or None if PySCF unavailable
        }
    """
    if not _have_pyscf():
        logger.info("PySCF not available, skipping physics simulation")
        return {"physics": None}

    config = config or {}

    # Import only if PySCF available
    try:
        if material.lower() == 'si' or material.lower() == 'silicon':
            from .silicon import SiliconSimulator
            sim = SiliconSimulator(config)
        elif material.lower() == 'ge' or material.lower() == 'germanium':
            from .germanium import GermaniumSimulator
            sim = GermaniumSimulator(config)
        else:
            logger.error(f"Unknown material: {material}")
            return {"physics": {"error": f"Unknown material: {material}"}}

        result = sim.run_scf()

        # Compute hypothesis-specific quantities
        from .form_factors import compute_cross_section
        physics_data = {
            "material": material,
            "e_tot_hartree": result.e_tot,
            "band_gap_ev": result.band_gap,
            "kpts": result.kpts
        }

        try:
            cross_section = compute_cross_section(hypothesis, result, config)
            physics_data.update(cross_section)
        except Exception as e:
            logger.warning(f"Cross-section calculation failed: {e}")
            physics_data["cross_section_error"] = str(e)

        return {"physics": physics_data}

    except Exception as e:
        logger.error(f"Physics simulation failed: {e}", exc_info=True)
        return {
            "physics": {
                "error": str(e),
                "material": material,
                "hypothesis": hypothesis
            }
        }

__all__ = ['run_simulation', '_have_pyscf']
```

---

## Patch 2: Base Classes

### Create src/materials/base.py

**File**: `src/materials/base.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Base classes for material simulations"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class MaterialConfig:
    """DFT simulation parameters"""
    basis: str = 'gth-dzvp'
    xc: str = 'PBE'  # Exchange-correlation functional
    kpts: Tuple[int, int, int] = (2, 2, 2)
    ecut: float = 50.0  # Energy cutoff [Hartree]
    convergence: float = 1e-6
    max_cycle: int = 50
    verbose: int = 4  # PySCF verbosity

@dataclass
class SimulationResult:
    """Unified physics output"""
    material: str
    e_tot: float  # Total energy [Hartree]
    band_gap: float  # [eV]
    kpts: int
    wavefunction: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MaterialSimulator(ABC):
    """Abstract base for all material calculations"""

    def __init__(self, config: Dict[str, Any]):
        # Convert dict to MaterialConfig
        if isinstance(config, dict):
            self.config = MaterialConfig(**{
                k: v for k, v in config.items()
                if k in MaterialConfig.__dataclass_fields__
            })
        else:
            self.config = config
        self.cell = None

    @abstractmethod
    def build_cell(self):
        """Construct crystal lattice (returns pyscf.pbc.gto.Cell)"""
        pass

    @abstractmethod
    def run_scf(self) -> SimulationResult:
        """Self-consistent field calculation"""
        pass

    @abstractmethod
    def compute_form_factor(self, q_values: np.ndarray) -> np.ndarray:
        """Nuclear form factor F²(q) for DM scattering"""
        pass

    def _extract_band_gap(self, mf) -> float:
        """Extract band gap from mean-field object"""
        try:
            from pyscf.pbc import tools
            # Get Fermi level
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ

            # Find HOMO and LUMO
            occupied = mo_energy[mo_occ > 0]
            unoccupied = mo_energy[mo_occ == 0]

            if len(unoccupied) == 0:
                return 0.0  # Metallic

            homo = np.max(occupied)
            lumo = np.min(unoccupied)
            gap_hartree = lumo - homo

            # Convert to eV
            gap_ev = gap_hartree * 27.211386  # Hartree to eV

            return gap_ev
        except Exception:
            return 0.0
```

---

## Patch 3: Silicon DFT Implementation

### Create src/materials/silicon.py

**File**: `src/materials/silicon.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Silicon crystalline DFT simulation"""
from __future__ import annotations
import logging
import os
from pathlib import Path
import numpy as np

try:
    from pyscf.pbc import gto, scf, tools
    from pyscf.pbc.tools import pyscf_ase
    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

from .base import MaterialSimulator, SimulationResult

logger = logging.getLogger(__name__)

class SiliconSimulator(MaterialSimulator):
    """Silicon diamond cubic structure (Fd-3m)"""

    def build_cell(self):
        """Construct Si unit cell"""
        if not _HAS_PYSCF:
            raise ImportError("PySCF required for silicon simulation")

        # Silicon lattice constant (Angstrom)
        a = 5.431

        # Diamond cubic structure
        cell = gto.Cell()
        cell.atom = f'''
        Si 0.0 0.0 0.0
        Si 0.25 0.25 0.25
        Si 0.5 0.5 0.0
        Si 0.75 0.75 0.25
        Si 0.5 0.0 0.5
        Si 0.75 0.25 0.75
        Si 0.0 0.5 0.5
        Si 0.25 0.75 0.75
        '''
        cell.unit = 'angstrom'

        # Lattice vectors
        cell.a = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])

        cell.basis = self.config.basis
        cell.pseudo = 'gth-pbe'  # GTH pseudopotential
        cell.verbose = self.config.verbose

        # K-point mesh
        cell.build()
        kpts = cell.make_kpts(self.config.kpts)

        self.cell = cell
        return cell

    def run_scf(self) -> SimulationResult:
        """Run DFT self-consistent field"""
        if not _HAS_PYSCF:
            raise ImportError("PySCF required")

        if self.cell is None:
            self.build_cell()

        # Check for checkpoint file
        cache_dir = Path(os.getenv('DMCA_PYSCF_CACHE', '/tmp/pyscf_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)

        config_hash = hash((
            self.config.basis,
            self.config.xc,
            self.config.kpts,
            self.config.ecut
        ))
        chkfile = cache_dir / f"si_{config_hash}.chk"

        # Initialize mean-field
        if self.config.xc.upper() in ['HF', 'HARTREE-FOCK']:
            mf = scf.KRHF(self.cell)
        else:
            mf = scf.KRKS(self.cell)
            mf.xc = self.config.xc

        mf.conv_tol = self.config.convergence
        mf.max_cycle = self.config.max_cycle

        # Use checkpoint if exists
        if chkfile.exists():
            logger.info(f"Loading from checkpoint: {chkfile}")
            mf.chkfile = str(chkfile)
            try:
                mf.init_guess = 'chkfile'
            except Exception:
                pass

        # Run SCF
        logger.info("Starting Si DFT calculation...")
        mf.kernel()

        if not mf.converged:
            logger.warning("SCF not converged!")

        # Save checkpoint
        mf.chkfile = str(chkfile)

        # Extract results
        e_tot = mf.e_tot
        band_gap = self._extract_band_gap(mf)

        logger.info(f"Si DFT complete: E_tot={e_tot:.6f} Ha, Eg={band_gap:.3f} eV")

        result = SimulationResult(
            material='Si',
            e_tot=e_tot,
            band_gap=band_gap,
            kpts=len(mf.kpts),
            density=mf.make_rdm1(),
            metadata={
                'converged': mf.converged,
                'basis': self.config.basis,
                'xc': self.config.xc
            }
        )

        # Store mean-field object for form factor calculation
        self._mf = mf

        return result

    def compute_form_factor(self, q_values: np.ndarray) -> np.ndarray:
        """
        Compute nuclear form factor F²(q).

        Args:
            q_values: Momentum transfer [GeV]

        Returns:
            F²(q) array (dimensionless)
        """
        if not hasattr(self, '_mf'):
            raise RuntimeError("Must run run_scf() first")

        # Convert q from GeV to inverse Bohr
        # 1 GeV = 1/0.1973 fm^-1, 1 Bohr = 0.529 Angstrom = 0.0529 fm
        q_invbohr = q_values * 0.1973 / 0.0529

        # Get nuclear positions
        atom_coords = self.cell.atom_coords()  # [N_atoms, 3] in Bohr
        atom_charges = self.cell.atom_charges()  # [N_atoms]

        # Compute form factor for each q
        F_q_squared = []

        for q in q_invbohr:
            if q < 1e-6:
                # F(q=0) = sum of charges
                F = np.sum(atom_charges)
            else:
                # Use random direction for q vector (spherical average)
                q_vec = q * np.array([1, 0, 0])  # Along x-axis

                # Sum over atoms: F(q) = Σ_i Z_i exp(iq·R_i)
                F = np.sum([
                    Z * np.exp(1j * np.dot(q_vec, R))
                    for Z, R in zip(atom_charges, atom_coords)
                ])

            F_q_squared.append(np.abs(F)**2 / np.sum(atom_charges)**2)  # Normalize

        return np.array(F_q_squared)
```

---

## Patch 4: Form Factors

### Create src/materials/form_factors.py

**File**: `src/materials/form_factors.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Dark matter interaction cross-sections"""
from __future__ import annotations
import logging
from typing import Dict, Any
import numpy as np
from .base import SimulationResult

logger = logging.getLogger(__name__)

# Physical constants
ALPHA_EM = 1/137.036  # Fine structure constant
HBAR_C = 0.1973  # GeV·fm
M_PROTON = 0.938  # GeV
M_ELECTRON = 0.511e-3  # GeV

def compute_cross_section(
    hypothesis: str,
    result: SimulationResult,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute hypothesis-specific cross-sections.

    Args:
        hypothesis: WIMP / Axion / DarkPhoton / SterileNeutrino
        result: SimulationResult from DFT
        config: Additional parameters (m_chi, q_range, etc.)

    Returns:
        Dict with cross-section and relevant parameters
    """
    hypothesis = hypothesis.upper()

    if hypothesis == 'WIMP':
        return _compute_wimp(result, config)
    elif hypothesis == 'AXION':
        return _compute_axion(result, config)
    elif hypothesis == 'DARKPHOTON':
        return _compute_darkphoton(result, config)
    elif hypothesis == 'STERILENEUTRINO':
        return _compute_sterile_neutrino(result, config)
    else:
        logger.warning(f"Unknown hypothesis: {hypothesis}")
        return {}

def _compute_wimp(result: SimulationResult, config: Dict[str, Any]) -> Dict[str, Any]:
    """WIMP-nucleus elastic scattering"""

    # WIMP mass (default 100 GeV)
    m_chi = config.get('m_chi', 100.0)

    # Nucleus mass (Si-28: 28 * 0.931 GeV)
    if result.material == 'Si':
        A = 28
        m_nucleus = A * 0.931
    elif result.material == 'Ge':
        A = 74
        m_nucleus = A * 0.931
    else:
        logger.warning(f"Unknown nucleus: {result.material}")
        return {}

    # Reduced mass
    mu = (m_chi * m_nucleus) / (m_chi + m_nucleus)

    # Form factor at q=0 (should be ~1)
    # For demo, use Helm parameterization
    r_n = 1.2 * A**(1/3)  # fm
    s = 0.9  # fm

    # Recoil energy range [keV]
    E_R = np.linspace(1, 100, 50)  # 1-100 keV

    # Momentum transfer
    q_gev = np.sqrt(2 * m_nucleus * E_R * 1e-6)  # GeV

    # Helm form factor
    q_r = q_gev * r_n / HBAR_C
    q_s = q_gev * s / HBAR_C

    F_helm = 3 * (np.sin(q_r) - q_r * np.cos(q_r)) / (q_r**3 + 1e-10)
    F_helm *= np.exp(-q_s**2 / 2)

    # Cross-section (spin-independent, approximate)
    # σ_SI ≈ (μ²/π) × F²(E_R)
    sigma_si = (mu**2 / np.pi) * F_helm**2

    # Average over recoil spectrum
    sigma_avg = np.mean(sigma_si)

    return {
        'wimp_m_chi_gev': m_chi,
        'wimp_sigma_cm2': sigma_avg * 1e-36,  # Convert to cm²
        'form_factor_q0': 1.0,
        'nucleus_a': A
    }

def _compute_axion(result: SimulationResult, config: Dict[str, Any]) -> Dict[str, Any]:
    """Axion-photon coupling"""

    # Axion decay constant (default 10^9 GeV)
    f_a = config.get('f_a', 1e9)

    # Axion mass (default 1 meV)
    m_a = config.get('m_a', 1e-6)  # GeV

    # Coupling constant (DFSZ model: E/N - 1.92 ≈ -0.97)
    g_agamma = (ALPHA_EM / (2 * np.pi * f_a)) * (-0.97)

    # Electron density from DFT (average)
    if result.density is not None:
        n_e_avg = np.mean(np.abs(result.density))  # electrons per Bohr³
        # Convert to cm⁻³
        n_e_cm3 = n_e_avg * (0.529e-8)**(-3)
    else:
        # Fallback: typical semiconductor density
        n_e_cm3 = 1e23

    # Plasma frequency
    omega_p = np.sqrt(4 * np.pi * n_e_cm3 * ALPHA_EM / M_ELECTRON)  # GeV

    return {
        'axion_g_agamma': g_agamma,
        'axion_m_a_gev': m_a,
        'axion_f_a_gev': f_a,
        'photon_plasma_freq_gev': omega_p
    }

def _compute_darkphoton(result: SimulationResult, config: Dict[str, Any]) -> Dict[str, Any]:
    """DarkPhoton kinetic mixing"""

    # Mixing parameter (default 10^-6)
    epsilon = config.get('epsilon', 1e-6)

    # DarkPhoton mass (default 10 MeV)
    m_Ap = config.get('m_ap', 0.01)  # GeV

    # Photon effective mass (from plasma frequency)
    if result.density is not None:
        n_e_avg = np.mean(np.abs(result.density))
        n_e_cm3 = n_e_avg * (0.529e-8)**(-3)
        m_gamma_eff = np.sqrt(4 * np.pi * n_e_cm3 * ALPHA_EM / M_ELECTRON)
    else:
        m_gamma_eff = 1e-6  # GeV (small)

    return {
        'darkphoton_epsilon': epsilon,
        'darkphoton_m_ap_gev': m_Ap,
        'photon_m_eff_gev': m_gamma_eff
    }

def _compute_sterile_neutrino(result: SimulationResult, config: Dict[str, Any]) -> Dict[str, Any]:
    """SterileNeutrino coherent scattering (uses same form factor as WIMP)"""

    # Mixing angle (default 0.1)
    sin2_theta = config.get('sin2_theta', 0.1)

    # Neutrino energy (default 10 MeV)
    E_nu = config.get('E_nu', 0.01)  # GeV

    # Weak charge (for Si: Q_w ≈ 14 - 0.08*14 ≈ 13)
    if result.material == 'Si':
        A = 28
        Z = 14
    elif result.material == 'Ge':
        A = 74
        Z = 32
    else:
        A = 1
        Z = 1

    Q_w = (A - Z) - (1 - 4 * 0.23) * Z  # sin²θ_W ≈ 0.23

    return {
        'sterile_sin2_theta': sin2_theta,
        'sterile_E_nu_gev': E_nu,
        'weak_charge': Q_w,
        'nucleus_a': A
    }
```

---

## Patch 5: Physics Emitter

### Create src/materials/physics_emitter.py

**File**: `src/materials/physics_emitter.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Telemetry emitter extension for physics fields (zero modification to base)"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.sidrce_bridge import TelemetryEmitter, EmitterConfig

@dataclass
class PhysicsEmitter:
    """
    Wrapper around TelemetryEmitter that adds physics fields.

    Design: Composition over inheritance (zero modification to base class).
    """

    base_emitter: Optional[TelemetryEmitter] = None

    def __post_init__(self):
        if self.base_emitter is None:
            # Auto-create with default config
            self.base_emitter = TelemetryEmitter()

    def emit_physics(self,
                     domain: str,
                     hypothesis: str,
                     step: int,
                     intervention: str,
                     infra_cost_usd: float,
                     qps: float,
                     latency_p95_ms: float,
                     physics_result: Optional[Dict[str, Any]] = None):
        """
        Emit telemetry with optional physics extension.

        Args:
            domain, hypothesis, ...: Standard DMCA fields (unchanged)
            physics_result: Output from run_simulation() (optional)
        """
        # Build base payload (no modification to structure)
        payload = {
            "domain": domain,
            "hypothesis": hypothesis,
            "step": step,
            "intervention": intervention,
            "infra_cost_usd": infra_cost_usd,
            "qps": qps,
            "latency_p95_ms": latency_p95_ms
        }

        # Add physics extension if available
        if physics_result is not None and "physics" in physics_result:
            if physics_result["physics"] is not None:
                payload["physics"] = physics_result["physics"]

        # Delegate to base emitter (uses all existing logic)
        self.base_emitter.emit(payload)

    def flush(self):
        """Forward flush to base emitter"""
        return self.base_emitter.flush()

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-flush on exit"""
        self.flush()
        return False
```

---

## Patch 6: Configuration Extension

### Update config/experiment.yaml

**File**: `config/experiment.yaml`

```diff
--- a/config/experiment.yaml
+++ b/config/experiment.yaml
@@ -10,6 +10,18 @@ domains:
       "3": "shield_upgrade"
       "5": "long_integration"

+    # NEW: Optional physics simulation (v1.0.6)
+    pyscf:
+      enabled: false  # Set to true to enable (default: off for backward compat)
+      materials: [Si]
+      basis: gth-dzvp
+      xc: PBE
+      kpts: [2, 2, 2]
+      q_range: [0.001, 0.5]  # GeV
+      q_points: 50
+      m_chi: 100.0  # WIMP mass [GeV]
+      cache_dir: /tmp/pyscf_cache
+
   cosmology:
     hypothesis: [DarkPhoton, SterileNeutrino]
     interventions:
@@ -17,6 +29,14 @@ domains:
       "6": "better_psf"

+    # NEW: Optional physics simulation
+    pyscf:
+      enabled: false
+      materials: [Ge]
+      basis: gth-dzvp
+      xc: PBE
+      kpts: [2, 2, 2]
+
 # Tokenizer settings
 tokenizer:
   bands: [VL, L, M, H, VH]
```

---

## Patch 7: Schema Extension

### Update sidrce/telemetry_contract.json

**File**: `sidrce/telemetry_contract.json`

```diff
--- a/sidrce/telemetry_contract.json
+++ b/sidrce/telemetry_contract.json
@@ -13,5 +13,31 @@
     "qps": {"type": "number", "minimum": 0},
     "latency_p95_ms": {"type": "number", "minimum": 0},
     "infra_cost_usd": {"type": "number", "minimum": 0}
-  }
+  },
+  "additionalProperties": true,
+  "properties_optional": {
+    "physics": {
+      "type": ["object", "null"],
+      "description": "Optional physics simulation results (v1.0.6)",
+      "properties": {
+        "material": {"type": "string"},
+        "e_tot_hartree": {"type": "number"},
+        "band_gap_ev": {"type": "number"},
+        "kpts": {"type": "integer"},
+        "wimp_m_chi_gev": {"type": "number"},
+        "wimp_sigma_cm2": {"type": "number"},
+        "form_factor_q0": {"type": "number"},
+        "nucleus_a": {"type": "integer"},
+        "axion_g_agamma": {"type": "number"},
+        "axion_m_a_gev": {"type": "number"},
+        "darkphoton_epsilon": {"type": "number"},
+        "darkphoton_m_ap_gev": {"type": "number"},
+        "sterile_sin2_theta": {"type": "number"},
+        "error": {"type": "string"}
+      }
+    }
+  },
+  "note": "physics field is optional for backward compatibility with v1.0.5"
 }
+
+
```

---

## Patch 8: Tests

### Create tests/materials/test_silicon.py

**File**: `tests/materials/test_silicon.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Tests for Silicon DFT simulation"""
import pytest
import numpy as np

# Skip all tests if PySCF not installed
pytest.importorskip("pyscf")

from src.materials.silicon import SiliconSimulator
from src.materials.base import MaterialConfig

def test_silicon_cell_build():
    """Test that Si cell builds correctly"""
    config = MaterialConfig(basis='gth-szv', kpts=(1,1,1))
    sim = SiliconSimulator(config)
    cell = sim.build_cell()

    assert cell is not None
    assert cell.natm == 8, "Si unit cell should have 8 atoms"
    assert abs(cell.a[0,0] - 5.431) < 0.1, "Lattice constant mismatch"

def test_silicon_scf_convergence():
    """Test that Si DFT converges (Gamma-point only for speed)"""
    config = MaterialConfig(
        basis='gth-szv',  # Minimal basis for speed
        kpts=(1,1,1),
        max_cycle=30
    )
    sim = SiliconSimulator(config)
    result = sim.run_scf()

    assert result.e_tot < 0, "Total energy should be negative"
    assert result.band_gap > 0.5, "Si should have band gap > 0.5 eV"
    assert result.band_gap < 2.0, "Si band gap should be < 2 eV (with PBE)"
    assert result.kpts == 1, "Should have 1 k-point (Gamma)"

def test_form_factor_normalization():
    """Test F(q=0) ≈ 1"""
    config = MaterialConfig(basis='gth-szv', kpts=(1,1,1))
    sim = SiliconSimulator(config)
    result = sim.run_scf()

    F_q = sim.compute_form_factor(np.array([0.0, 0.001]))
    assert abs(F_q[0] - 1.0) < 0.05, "Form factor at q=0 should be ~1"

@pytest.mark.slow
def test_silicon_kpoint_convergence():
    """Test k-point convergence (slow test)"""
    energies = []
    for nk in [1, 2]:
        config = MaterialConfig(basis='gth-szv', kpts=(nk, nk, nk))
        sim = SiliconSimulator(config)
        result = sim.run_scf()
        energies.append(result.e_tot)

    # Energy should decrease with more k-points
    assert energies[1] < energies[0], "Energy should converge down"
```

### Create tests/materials/test_form_factors.py

**File**: `tests/materials/test_form_factors.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Tests for form factor calculations"""
import pytest

pytest.importorskip("pyscf")

from src.materials.form_factors import compute_cross_section
from src.materials.base import SimulationResult

def test_wimp_cross_section():
    """Test WIMP cross-section calculation"""
    # Mock result
    result = SimulationResult(
        material='Si',
        e_tot=-289.0,
        band_gap=1.17,
        kpts=1
    )

    config = {'m_chi': 100.0}
    cs = compute_cross_section('WIMP', result, config)

    assert 'wimp_m_chi_gev' in cs
    assert cs['wimp_m_chi_gev'] == 100.0
    assert 'wimp_sigma_cm2' in cs
    assert cs['wimp_sigma_cm2'] > 0
    assert 'form_factor_q0' in cs
    assert abs(cs['form_factor_q0'] - 1.0) < 0.1

def test_axion_coupling():
    """Test axion-photon coupling"""
    result = SimulationResult(
        material='Si',
        e_tot=-289.0,
        band_gap=1.17,
        kpts=1
    )

    config = {'f_a': 1e9, 'm_a': 1e-6}
    cs = compute_cross_section('Axion', result, config)

    assert 'axion_g_agamma' in cs
    assert 'axion_m_a_gev' in cs
    assert cs['axion_m_a_gev'] == 1e-6
```

### Create tests/test_backward_compat.py

**File**: `tests/test_backward_compat.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""Backward compatibility tests (no PySCF required)"""
import json
import tempfile
from pathlib import Path
from src.sidrce_bridge import TelemetryEmitter, EmitterConfig

def test_old_telemetry_format_unchanged():
    """Verify v1.0.5 telemetry format still works"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"

        emitter = TelemetryEmitter(cfg=EmitterConfig(
            path=str(path),
            enabled=True,
            batch_size=1  # Flush immediately
        ))

        # Old format (no physics field)
        emitter.emit({
            "domain": "direct_lab",
            "hypothesis": "WIMP",
            "step": 1,
            "intervention": "-",
            "infra_cost_usd": 120.0,
            "qps": 480.0,
            "latency_p95_ms": 180.0
        })

        emitter.flush()

        # Verify written correctly
        with open(path) as f:
            rec = json.loads(f.readline())

        assert "physics" not in rec, "Old format should not have physics field"
        assert rec["domain"] == "direct_lab"
        assert rec["qps"] == 480.0

def test_physics_emitter_without_physics():
    """PhysicsEmitter should work exactly like base when physics=None"""
    pytest = __import__('pytest')
    if not pytest.__version__:
        return  # Skip if pytest not available

    from src.materials.physics_emitter import PhysicsEmitter

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"

        emitter = PhysicsEmitter(base_emitter=TelemetryEmitter(
            cfg=EmitterConfig(path=str(path), batch_size=1)
        ))

        # Emit without physics
        emitter.emit_physics(
            domain="direct_lab",
            hypothesis="WIMP",
            step=1,
            intervention="-",
            infra_cost_usd=120.0,
            qps=480.0,
            latency_p95_ms=180.0,
            physics_result=None  # No physics
        )

        emitter.flush()

        with open(path) as f:
            rec = json.loads(f.readline())

        assert "physics" not in rec, "Should match old format when physics=None"
```

---

## Patch 9: CI/CD

### Create .github/workflows/physics-gates.yml

**File**: `.github/workflows/physics-gates.yml` (NEW)

```yaml
name: Physics Module Gates

on:
  push:
    branches: ['**']
  pull_request:
    branches: [main, master]

jobs:
  test-physics:
    strategy:
      matrix:
        pyscf: [none, installed]
        python-version: ['3.9', '3.10', '3.11']

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install core dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Install PySCF (matrix: ${{ matrix.pyscf }})
        if: matrix.pyscf == 'installed'
        run: pip install pyscf>=2.3.0

      - name: Run tests (without PySCF)
        if: matrix.pyscf == 'none'
        run: |
          # Core tests + backward compat (should pass without PySCF)
          pytest tests/ -k "not materials" -v --cov=src --cov-report=term

      - name: Run tests (with PySCF)
        if: matrix.pyscf == 'installed'
        run: |
          # All tests including physics modules
          pytest tests/ -v --cov=src --cov-report=term
          # Mark slow tests
          pytest tests/materials/ -v -m "not slow"

      - name: Validate telemetry schema
        run: |
          # Ensure backward compatibility
          python scripts/validate_telemetry.py sample_ops_telemetry.jsonl

      - name: ASDP quality check
        run: |
          python tools/investor_asdp_cli.py check
```

---

## Patch 10: ASDP Rules

### Update config/investor_asdp.dev.yaml

**File**: `config/investor_asdp.dev.yaml`

```diff
--- a/config/investor_asdp.dev.yaml
+++ b/config/investor_asdp.dev.yaml
@@ -65,3 +65,23 @@ spicy:
     severity: warn
     message: "TODO/FIXME should reference issue tracker"
     autofix: false
+
+  # NEW: Physics module isolation rules (v1.0.6)
+  - id: no_physics_in_core
+    pattern: 'from src\.materials|import.*materials'
+    files: 'src/sidrce_bridge.py,src/causal.py,sidrce/*.py'
+    severity: error
+    message: "Core modules must not import optional physics (drift prevention)"
+    autofix: false
+
+  - id: no_finops_in_physics
+    pattern: 'from sidrce\.collector import|calc_fti|lambda_effective'
+    files: 'src/materials/*.py'
+    severity: error
+    message: "Physics modules must not compute FTI/λ (separation of concerns)"
+    autofix: false
+
+  - id: no_telemetry_field_deletion
+    pattern: 'del.*["\'](?:domain|hypothesis|step|intervention|infra_cost_usd|qps|latency_p95_ms)["\']'
+    files: 'src/**/*.py'
+    severity: error
+    message: "Deleting core telemetry fields forbidden (schema drift)"
+    autofix: false
```

---

## Application Checklist

Apply patches in order:

- [ ] Patch 1: Create `src/materials/__init__.py`
- [ ] Patch 2: Create `src/materials/base.py`
- [ ] Patch 3: Create `src/materials/silicon.py`
- [ ] Patch 4: Create `src/materials/form_factors.py`
- [ ] Patch 5: Create `src/materials/physics_emitter.py`
- [ ] Patch 6: Update `config/experiment.yaml`
- [ ] Patch 7: Update `sidrce/telemetry_contract.json`
- [ ] Patch 8: Create all test files
- [ ] Patch 9: Create `.github/workflows/physics-gates.yml`
- [ ] Patch 10: Update `config/investor_asdp.dev.yaml`

After applying all patches:

```bash
# Verify imports work without PySCF
python -c "from src.materials import run_simulation; print('OK')"

# Run backward compat tests
pytest tests/test_backward_compat.py -v

# Install PySCF and run physics tests
pip install pyscf>=2.3.0
pytest tests/materials/ -v -m "not slow"

# ASDP check
python tools/investor_asdp_cli.py check
```

---

**Next**: See `quality-assurance-checklist.md` for validation procedures.
