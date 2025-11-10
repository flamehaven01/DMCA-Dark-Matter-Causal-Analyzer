# PySCF Integration Technical Guide
## DMCA v1.0.6 - Quantum Chemistry Module for Dark Matter Simulations

**Document Version**: 1.0
**Date**: 2025-11-10
**Target Audience**: Physics PhDs, Quantum Chemistry Engineers, Senior Developers
**Prerequisites**: PySCF 2.3+, NumPy, SciPy, Solid-state physics background

---

## Executive Summary

This document provides **complete technical specifications** for integrating PySCF-based quantum chemistry simulations into the DMCA (Dark Matter Causal Analyzer) framework. The integration enables:

1. **Crystal form factor calculations** for DM-electron scattering (Si, Ge, GaAs)
2. **WIMP direct detection** cross-section predictions
3. **Axion-photon coupling** in semiconductor crystals
4. **DarkPhoton mixing parameters** via band structure analysis
5. **SterileNeutrino coherent scattering** form factors

**Design Principle**: Zero drift from existing DMCA architecture. All physics modules are **optional plugins** that extend, not modify, core telemetry/FinOps systems.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Physics Background](#2-physics-background)
3. [Integration Points](#3-integration-points)
4. [Module Structure](#4-module-structure)
5. [API Specifications](#5-api-specifications)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Quality Assurance](#7-quality-assurance)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

**Zero-Drift Principle**: Physics modules MUST NOT:
- Modify existing `src/sidrce_bridge.py` telemetry emitter
- Change `sidrce/collector.py` FTI/λ calculations
- Alter `src/causal.py` inference logic
- Break existing CI/CD gates

**Integration Strategy**: Plugin architecture with:
- Isolated namespace: `src/materials/`
- Optional dependency: PySCF installation not required for DMCA core
- Telemetry extension: New fields added, not replaced
- Backward compatibility: All existing tests pass without PySCF

### 1.2 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DMCA Core (v1.0.5)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ sidrce_bridge│  │   causal.py  │  │  metrics.py  │     │
│  │  (telemetry) │  │  (inference) │  │  (REMOVED)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ extends (no modification)
                           │
┌─────────────────────────────────────────────────────────────┐
│              Physics Module (v1.0.6 - NEW)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              src/materials/                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │ silicon.py │  │ germanium. │  │  gaas.py   │    │  │
│  │  │  (Si DFT)  │  │   py (Ge)  │  │ (GaAs DFT) │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐    │  │
│  │  │      form_factors.py                       │    │  │
│  │  │  - WIMP cross-sections                     │    │  │
│  │  │  - Axion coupling                          │    │  │
│  │  │  - DarkPhoton mixing                       │    │  │
│  │  │  - SterileNeutrino coherent scattering     │    │  │
│  │  └────────────────────────────────────────────┘    │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐    │  │
│  │  │      physics_emitter.py                    │    │  │
│  │  │  - Wraps TelemetryEmitter                  │    │  │
│  │  │  - Adds physics fields (E_tot, q_scan)     │    │  │
│  │  └────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ▼
                  SIDRCE (unchanged)
                     FinOps + Grafana
```

### 1.3 Data Flow

```
Experiment Config (config/experiment.yaml)
   └─> hypothesis: WIMP / Axion / DarkPhoton / SterileNeutrino
         │
         ▼
   Material Module (src/materials/silicon.py)
         │
         ├─> PySCF DFT Calculation
         │    - Band structure
         │    - Density of states
         │    - Wavefunctions
         │
         ├─> Form Factor Calculator (form_factors.py)
         │    - Momentum transfer q scan
         │    - Cross-section σ(E_R)
         │    - Coupling constants
         │
         ▼
   PhysicsEmitter (physics_emitter.py)
         │
         ├─> Extends telemetry with:
         │    {"e_tot": -289.xyz, "sigma_wimp": 1e-40, "q_max": 0.5}
         │
         ▼
   ops_telemetry.jsonl (SIDRCE ingestion)
         │
         ▼
   FTI/λ Calculation + Grafana (unchanged)
```

---

## 2. Physics Background

### 2.1 WIMP Direct Detection

**Physical Process**: WIMP-nucleus elastic scattering
```
χ + (Z, A) → χ + (Z, A)*
```

**Differential Rate**:
```
dR/dE_R = (ρ_χ σ_χN)/(2 m_χ m_N) × F²(E_R) × ∫ v f(v) dv
```

Where:
- `ρ_χ = 0.3 GeV/cm³` (local DM density)
- `σ_χN` = WIMP-nucleon cross-section
- `F²(E_R)` = **Nuclear form factor** (PySCF computes this)
- `f(v)` = Maxwell-Boltzmann velocity distribution

**Form Factor Calculation** (PySCF role):
```python
F²(q) = |∫ d³r ρ(r) e^(iq·r)|²
```
- `ρ(r)` = Nuclear density from DFT calculation
- `q = √(2 m_N E_R)` = Momentum transfer

### 2.2 Axion-Photon Coupling

**Primakoff Process**: Axion ↔ Photon conversion in crystal
```
a + γ → γ  (in presence of E/B field)
```

**Coupling Strength**:
```
g_aγγ = (α/(2π f_a)) × (E/N - 1.92)
```

**PySCF Contribution**: Calculate crystal electric field `E_crystal` from band structure
```python
E_crystal = -∇V_Hartree(r)  # from self-consistent field
```

### 2.3 DarkPhoton Mixing

**Kinetic Mixing**: DarkPhoton A' mixes with SM photon
```
L_mix = (ε/2) F^μν F'_μν
```

**Mass Matrix**:
```
M² = [ m_γ²,     ε m_γ m_A' ]
     [ ε m_γ m_A',   m_A'²    ]
```

**PySCF Contribution**: Photon effective mass in medium
```python
m_γ² = ω_plasma² = 4π n_e α  # from band structure n_e
```

### 2.4 SterileNeutrino Coherent Scattering

**Process**: ν_s + nucleus → ν_s + nucleus
```
dσ/dE_R = (G_F² m_N)/(2π) × [1 - (m_N E_R)/(2 E_ν²)] × F²(q)
```

**PySCF Contribution**: Nuclear form factor `F²(q)` same as WIMP case

---

## 3. Integration Points

### 3.1 Telemetry Schema Extension

**Current Schema** (`sidrce/telemetry_contract.json`):
```json
{
  "domain": "direct_lab",
  "hypothesis": "WIMP",
  "step": 1,
  "intervention": "-",
  "infra_cost_usd": 120.0,
  "qps": 480.0,
  "latency_p95_ms": 180.0
}
```

**Extended Schema** (v1.0.6 - backward compatible):
```json
{
  "domain": "direct_lab",
  "hypothesis": "WIMP",
  "step": 1,
  "intervention": "-",
  "infra_cost_usd": 120.0,
  "qps": 480.0,
  "latency_p95_ms": 180.0,

  // NEW FIELDS (optional - only if PySCF available)
  "physics": {
    "material": "Si",
    "e_tot_hartree": -289.123456,
    "band_gap_ev": 1.17,
    "kpts": 8,
    "q_scan_points": 50,
    "wimp_sigma_cm2": 1.2e-40,
    "form_factor_q0": 0.98,
    "computation_time_sec": 145.3
  }
}
```

**Schema Validation Update**:
```diff
--- a/sidrce/telemetry_contract.json
+++ b/sidrce/telemetry_contract.json
@@ -15,6 +15,30 @@
     "qps": {"type": "number", "minimum": 0},
     "latency_p95_ms": {"type": "number", "minimum": 0},
     "infra_cost_usd": {"type": "number", "minimum": 0}
+  },
+  "properties_optional": {
+    "physics": {
+      "type": "object",
+      "properties": {
+        "material": {"type": "string"},
+        "e_tot_hartree": {"type": "number"},
+        "band_gap_ev": {"type": "number"},
+        "kpts": {"type": "integer"},
+        "q_scan_points": {"type": "integer"},
+        "wimp_sigma_cm2": {"type": "number"},
+        "axion_coupling": {"type": "number"},
+        "darkphoton_epsilon": {"type": "number"},
+        "form_factor_q0": {"type": "number"},
+        "computation_time_sec": {"type": "number"}
+      }
+    }
   }
 }
```

### 3.2 Configuration Extension

**config/experiment.yaml** (current):
```yaml
domains:
  direct_lab:
    hypothesis: [WIMP, Axion]
    interventions:
      "3": "shield_upgrade"
```

**Extended** (v1.0.6):
```yaml
domains:
  direct_lab:
    hypothesis: [WIMP, Axion]
    interventions:
      "3": "shield_upgrade"

    # NEW: Physics simulation parameters
    pyscf:
      enabled: true
      materials: [Si, Ge]  # compute both
      basis: gth-dzvp
      xc: PBE
      kpts: [2, 2, 2]  # k-point mesh
      q_range: [0.001, 0.5]  # momentum transfer [GeV]
      q_points: 50
```

### 3.3 Dependency Management

**requirements.txt** (already prepared):
```python
# Optional physics module
# pyscf>=2.3.0  # Uncomment to enable
```

**Runtime Detection** (similar to `src/causal.py`):
```python
def _have_pyscf() -> bool:
    try:
        import pyscf  # noqa
        return True
    except ImportError:
        return False

def run_simulation(material: str, hypothesis: str):
    if _have_pyscf():
        return _pyscf_simulation(material, hypothesis)
    else:
        # Graceful degradation: return mock results
        return _mock_simulation(material, hypothesis)
```

**Environment Variable Guard**:
```bash
# Enable physics simulations
export DMCA_PHYSICS_ENABLED=1
export DMCA_PYSCF_CACHE=/tmp/pyscf_cache  # DFT checkpoint dir
```

---

## 4. Module Structure

### 4.1 Directory Layout

```
src/materials/
├── __init__.py           # Module entry point + registry
├── base.py               # Abstract MaterialSimulator class
├── silicon.py            # Si crystalline DFT
├── germanium.py          # Ge crystalline DFT
├── gaas.py               # GaAs (future)
├── form_factors.py       # DM interaction cross-sections
├── physics_emitter.py    # Telemetry extension wrapper
└── utils/
    ├── __init__.py
    ├── crystal_builder.py   # Lattice construction helpers
    ├── band_analysis.py     # DOS, band gap extraction
    └── momentum_scan.py     # q-space integration

tests/materials/
├── test_silicon.py
├── test_form_factors.py
└── fixtures/
    └── si_reference.json    # Known-good results

docs/
├── pyscf-integration-technical-guide.md  (this file)
├── physics-simulation-mathematics.md
├── architecture-integration-design.md
├── implementation-patches.md
└── quality-assurance-checklist.md
```

### 4.2 Class Hierarchy

```python
# src/materials/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class MaterialConfig:
    """DFT simulation parameters"""
    basis: str = 'gth-dzvp'
    xc: str = 'PBE'  # Exchange-correlation functional
    kpts: tuple = (2, 2, 2)
    ecut: float = 50.0  # Hartree
    convergence: float = 1e-6

@dataclass
class SimulationResult:
    """Unified physics output"""
    material: str
    e_tot: float  # Total energy [Hartree]
    band_gap: float  # [eV]
    kpts: int
    wavefunction: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class MaterialSimulator(ABC):
    """Abstract base for all material calculations"""

    def __init__(self, config: MaterialConfig):
        self.config = config
        self.cell = None  # PySCF Cell object

    @abstractmethod
    def build_cell(self) -> 'pyscf.pbc.gto.Cell':
        """Construct crystal lattice"""
        pass

    @abstractmethod
    def run_scf(self) -> SimulationResult:
        """Self-consistent field calculation"""
        pass

    @abstractmethod
    def compute_form_factor(self, q_values: np.ndarray) -> np.ndarray:
        """F²(q) for DM scattering"""
        pass
```

---

## 5. API Specifications

### 5.1 High-Level Interface

```python
from src.materials import run_dm_simulation

# Simple API for hypothesis testing
result = run_dm_simulation(
    hypothesis='WIMP',
    material='Si',
    config={
        'basis': 'gth-dzvp',
        'kpts': (2, 2, 2),
        'q_max': 0.5  # GeV
    }
)

print(result)
# {
#   'e_tot': -289.123,
#   'band_gap_ev': 1.17,
#   'wimp_sigma_cm2': 1.2e-40,
#   'form_factor': array([...]),
#   'q_values': array([...])
# }
```

### 5.2 Telemetry Integration

```python
from src.sidrce_bridge import TelemetryEmitter
from src.materials.physics_emitter import PhysicsEmitter

# Wrap existing emitter (zero modification to original class)
emitter = PhysicsEmitter(base_emitter=TelemetryEmitter())

# Emit with physics extension
emitter.emit_physics(
    domain='direct_lab',
    hypothesis='WIMP',
    step=1,
    intervention='-',
    infra_cost_usd=120.0,
    qps=480.0,
    latency_p95_ms=180.0,
    # Physics fields automatically added
    physics_result=result
)
```

### 5.3 Batch Processing API

```python
from src.materials import batch_simulate

# Simulate multiple materials/hypotheses
results = batch_simulate(
    hypotheses=['WIMP', 'Axion'],
    materials=['Si', 'Ge'],
    config_template={'basis': 'gth-dzvp'},
    parallel=True,  # use multiprocessing
    emit_telemetry=True  # auto-emit to ops_telemetry.jsonl
)
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Deliverables**:
- [ ] `src/materials/__init__.py` (module registry)
- [ ] `src/materials/base.py` (abstract classes)
- [ ] `src/materials/utils/crystal_builder.py` (lattice helpers)
- [ ] Unit tests for base classes
- [ ] CI gate: tests pass without PySCF installed

**Acceptance Criteria**:
- Import `src.materials` succeeds even without PySCF
- All existing tests still pass
- ASDP Ω score ≥ 0.8

### Phase 2: Silicon DFT (Week 2)

**Deliverables**:
- [ ] `src/materials/silicon.py` (full DFT implementation)
- [ ] Band structure calculation
- [ ] Density of states
- [ ] Wavefunction export
- [ ] Integration tests with PySCF
- [ ] Reference data fixtures

**Acceptance Criteria**:
- Band gap within 5% of experimental (1.17 eV)
- Total energy convergence < 1e-6 Hartree
- Computation time < 5 min (8 k-points)

### Phase 3: Form Factors (Week 3)

**Deliverables**:
- [ ] `src/materials/form_factors.py`
- [ ] WIMP cross-section calculator
- [ ] Axion coupling extractor
- [ ] DarkPhoton mixing parameters
- [ ] SterileNeutrino form factors
- [ ] Validation against published results

**Acceptance Criteria**:
- WIMP σ within factor 2 of literature (Essig et al. 2016)
- Form factor F²(q=0) = 1 (normalization check)

### Phase 4: Telemetry Integration (Week 4)

**Deliverables**:
- [ ] `src/materials/physics_emitter.py`
- [ ] Schema extension (`telemetry_contract.json`)
- [ ] Backward compatibility tests
- [ ] Grafana dashboard panel for physics metrics
- [ ] Documentation updates

**Acceptance Criteria**:
- Old telemetry format still validates
- SIDRCE collector ingests extended schema
- No breakage in existing CI/CD

### Phase 5: Production Hardening (Week 5)

**Deliverables**:
- [ ] Performance profiling + optimization
- [ ] Checkpoint/restart for long runs
- [ ] Error handling + graceful degradation
- [ ] Multi-material batch processing
- [ ] User guide + tutorials

**Acceptance Criteria**:
- ASDP Ω score ≥ 0.85
- Test coverage ≥ 80%
- Zero regression in existing metrics

---

## 7. Quality Assurance

### 7.1 Test Coverage Requirements

| Component | Coverage | Strategy |
|-----------|----------|----------|
| `base.py` | 100% | Unit tests (mocked PySCF) |
| `silicon.py` | ≥90% | Integration tests (real PySCF) |
| `form_factors.py` | ≥85% | Physics validation fixtures |
| `physics_emitter.py` | 100% | Unit + integration tests |

### 7.2 CI/CD Matrix Extension

```yaml
# .github/workflows/physics-gates.yml
strategy:
  matrix:
    pyscf: [none, installed]
    python: [3.9, 3.10, 3.11]
```

**Gates**:
1. **Without PySCF**: All imports succeed, graceful degradation
2. **With PySCF**: Physics tests pass, reference results match
3. **Telemetry**: Extended schema validates, backward compatibility

### 7.3 Physics Validation

**Reference Calculations** (must match within tolerance):
```python
# tests/materials/fixtures/si_reference.json
{
  "material": "Si",
  "source": "VASP calculation (Essig et al. 2016)",
  "e_tot_hartree": -289.123456,
  "band_gap_ev": 1.17,
  "tolerance": 0.05,
  "form_factor_q0": 1.0,
  "form_factor_q05": 0.78
}
```

**Validation Script**:
```bash
python tests/materials/validate_physics.py --material Si --reference si_reference.json
# Output: PASS (band_gap: 1.18 eV, within 5% of reference 1.17 eV)
```

### 7.4 ASDP Integration

**config/investor_asdp.dev.yaml** extension:
```yaml
spicy:
  - id: physics_imports
    pattern: 'import pyscf'
    files: 'src/core/*'
    severity: error
    message: "PySCF imports not allowed in core modules (use src/materials only)"

  - id: telemetry_schema_drift
    pattern: 'del.*["'](domain|hypothesis|step|intervention|infra_cost_usd|qps|latency_p95_ms)["']'
    files: 'src/**/*.py'
    severity: error
    message: "Deleting core telemetry fields forbidden (drift prevention)"
```

---

## 8. Troubleshooting

### 8.1 PySCF Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'pyscf'`

**Solution**:
```bash
pip install pyscf>=2.3.0
# Or with conda:
conda install -c pyscf pyscf
```

**Problem**: `libxc` compilation error on macOS

**Solution**:
```bash
brew install libxc
export LDFLAGS="-L/opt/homebrew/opt/libxc/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libxc/include"
pip install pyscf --no-binary pyscf
```

### 8.2 Performance Issues

**Problem**: DFT calculation takes > 1 hour

**Solution**: Reduce k-point mesh
```yaml
pyscf:
  kpts: [1, 1, 1]  # Gamma-point only (fast but less accurate)
```

**Problem**: Out of memory

**Solution**: Enable disk-based density fitting
```python
from pyscf.pbc import df
mf.with_df = df.DF(cell)
mf.with_df._cderi_to_save = '/tmp/pyscf_cderi.h5'
```

### 8.3 Physics Validation Failures

**Problem**: Band gap off by >10%

**Diagnosis**:
```python
# Check k-point convergence
for kpts in [(1,1,1), (2,2,2), (4,4,4)]:
    result = run_scf(kpts=kpts)
    print(f"kpts={kpts}: Eg={result.band_gap:.3f} eV")
```

**Solution**: Increase k-point mesh or use hybrid functional (HSE06)

**Problem**: Form factor diverges at low q

**Solution**: Check momentum transfer units (GeV vs. inverse Bohr)
```python
q_gev = np.linspace(0.001, 0.5, 50)  # GeV
q_invbohr = q_gev * 0.1973 / 0.529  # convert to PySCF units
```

---

## Next Steps

1. Read companion documents:
   - `physics-simulation-mathematics.md` - Detailed derivations
   - `architecture-integration-design.md` - Code integration specs
   - `implementation-patches.md` - Ready-to-apply diff patches

2. Set up development environment:
   ```bash
   pip install pyscf pytest-cov
   export DMCA_PHYSICS_ENABLED=1
   ```

3. Run baseline tests:
   ```bash
   pytest tests/materials/ -v
   ```

4. Follow implementation patches in order (Phase 1 → Phase 5)

5. Validate with ASDP:
   ```bash
   python tools/investor_asdp_cli.py check
   ```

---

**Questions?** Open a GitHub issue or contact the physics team.

**Version History**:
- v1.0 (2025-11-10): Initial technical guide
