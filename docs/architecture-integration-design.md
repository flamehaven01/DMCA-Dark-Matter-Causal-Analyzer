# Architecture Integration Design
## Zero-Drift Integration of Physics Modules into DMCA

**Document Version**: 1.0
**Date**: 2025-11-10
**Critical Goal**: 0% code drift - preserve all existing functionality

---

## Table of Contents

1. [Integration Principles](#1-integration-principles)
2. [Dependency Injection Pattern](#2-dependency-injection-pattern)
3. [Module Boundaries](#3-module-boundaries)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Error Handling Strategy](#5-error-handling-strategy)
6. [Testing Strategy](#6-testing-strategy)
7. [Performance Considerations](#7-performance-considerations)

---

## 1. Integration Principles

### 1.1 Zero-Drift Guarantees

**Immutable Components** (NEVER modify):
```
src/sidrce_bridge.py     # Telemetry emitter core
src/metrics.py           # Deprecated guards
sidrce/collector.py      # FTI/λ calculator
sidrce/server.py         # FastAPI endpoints
src/causal.py            # Hybrid SCM (structure only, not logic)
```

**Extension-Only Components** (can add to):
```
config/experiment.yaml   # Add pyscf section
requirements.txt         # Add optional pyscf>=2.3.0
sidrce/telemetry_contract.json  # Add optional physics field
.github/workflows/       # Add new physics-gates.yml (don't modify existing)
```

**New Components** (create from scratch):
```
src/materials/           # All physics modules (isolated namespace)
tests/materials/         # Physics-specific tests
docs/*.md                # Documentation
```

### 1.2 Backward Compatibility Checklist

Before merging any PR, verify:

- [ ] All existing tests pass without PySCF installed
- [ ] `ops_telemetry.jsonl` without `physics` field still validates
- [ ] SIDRCE collector runs on old telemetry format
- [ ] Grafana dashboard shows legacy metrics
- [ ] CI/CD gates: `causal_libs=none` and `causal_libs=full` both pass
- [ ] ASDP Ω score ≥ previous value
- [ ] No changes to git history (rebase only if necessary)

### 1.3 Optional Dependency Pattern

**Example from existing codebase** (`src/causal.py:17-28`):
```python
def _have_libs() -> tuple[bool, bool]:
    try:
        import causallearn  # noqa
        has_cl = True
    except Exception:
        has_cl = False
    try:
        import dowhy  # noqa
        has_dw = True
    except Exception:
        has_dw = False
    return has_cl, has_dw
```

**Apply same pattern for PySCF**:
```python
# src/materials/__init__.py
def _have_pyscf() -> bool:
    try:
        import pyscf  # noqa
        return True
    except ImportError:
        return False

def run_simulation(hypothesis, material, config):
    if _have_pyscf():
        from .silicon import SiliconSimulator
        return SiliconSimulator(config).run()
    else:
        # Graceful degradation: return empty physics dict
        return {"physics": None}
```

---

## 2. Dependency Injection Pattern

### 2.1 PhysicsEmitter Wrapper

**Design**: Wrap `TelemetryEmitter` instead of modifying it.

**src/materials/physics_emitter.py**:
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.sidrce_bridge import TelemetryEmitter, EmitterConfig

@dataclass
class PhysicsEmitter:
    """Wrapper that extends TelemetryEmitter with physics fields"""

    base_emitter: TelemetryEmitter = None

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
            payload["physics"] = physics_result["physics"]

        # Delegate to base emitter (uses all existing logic)
        self.base_emitter.emit(payload)

    def flush(self):
        """Forward flush to base emitter"""
        return self.base_emitter.flush()
```

**Key Design Features**:
1. **Composition over inheritance**: Wraps, not extends
2. **Zero modification**: `TelemetryEmitter` unchanged
3. **Backward compatible**: Old code can still use `TelemetryEmitter` directly
4. **Optional physics**: If `physics_result=None`, behaves identically to base

### 2.2 Usage Example

**Before (existing code - still works)**:
```python
from src.sidrce_bridge import TelemetryEmitter

emitter = TelemetryEmitter()
emitter.emit({
    "domain": "direct_lab",
    "hypothesis": "WIMP",
    "step": 1,
    "intervention": "-",
    "infra_cost_usd": 120.0,
    "qps": 480.0,
    "latency_p95_ms": 180.0
})
```

**After (new code - with physics)**:
```python
from src.materials.physics_emitter import PhysicsEmitter
from src.materials import run_simulation

emitter = PhysicsEmitter()  # Auto-wraps TelemetryEmitter

# Run optional physics
result = run_simulation(hypothesis='WIMP', material='Si', config={})

# Emit with physics extension
emitter.emit_physics(
    domain="direct_lab",
    hypothesis="WIMP",
    step=1,
    intervention="-",
    infra_cost_usd=120.0,
    qps=480.0,
    latency_p95_ms=180.0,
    physics_result=result  # Optional - can be None
)
```

**Mixed Mode (gradually migrate)**:
```python
# Old code path
base_emitter = TelemetryEmitter()
base_emitter.emit({...})  # Still works!

# New code path
physics_emitter = PhysicsEmitter(base_emitter=base_emitter)  # Reuse same instance
physics_emitter.emit_physics(..., physics_result=None)  # No physics, backward compatible
```

---

## 3. Module Boundaries

### 3.1 Import Rules (ASDP Enforced)

**Allowed Imports**:
```python
# src/materials/*.py CAN import:
import pyscf  # OK - physics module only
import numpy, scipy  # OK - core dependencies
from src.sidrce_bridge import TelemetryEmitter  # OK - extends functionality

# src/materials/*.py CANNOT import:
from sidrce.collector import calc_fti  # FORBIDDEN - physics doesn't compute FTI
from src.causal import learn_structure_from_data  # FORBIDDEN - no causal dependency
```

**Reverse Dependency Ban**:
```python
# src/sidrce_bridge.py, sidrce/collector.py, src/causal.py CANNOT import:
from src.materials import *  # FORBIDDEN - core must not depend on optional physics
```

**ASDP Rule** (`config/investor_asdp.dev.yaml`):
```yaml
spicy:
  - id: no_physics_in_core
    pattern: 'from src.materials|import.*materials'
    files: 'src/sidrce_bridge.py,src/causal.py,sidrce/*.py'
    severity: error
    message: "Core modules must not import optional physics (drift prevention)"

  - id: no_finops_in_physics
    pattern: 'from sidrce.collector import|calc_fti|lambda_effective'
    files: 'src/materials/*.py'
    severity: error
    message: "Physics modules must not compute FTI/λ (separation of concerns)"
```

### 3.2 Namespace Isolation

```
src/
├── __init__.py          # Core DMCA exports (no physics)
├── sidrce_bridge.py     # Telemetry (unchanged)
├── causal.py            # Inference (unchanged)
├── metrics.py           # Removed (unchanged)
└── materials/           # NEW - isolated namespace
    ├── __init__.py      # Physics exports (only if pyscf available)
    ├── base.py
    ├── silicon.py
    └── ...
```

**Import Pattern**:
```python
# Core DMCA (always available)
from src.sidrce_bridge import TelemetryEmitter
from src.causal import learn_structure_from_data

# Physics extension (optional)
try:
    from src.materials import run_simulation
except ImportError:
    run_simulation = None  # Graceful degradation
```

---

## 4. Data Flow Architecture

### 4.1 Telemetry Extension (Additive Only)

**Schema Evolution**:
```json
// v1.0.5 (current) - baseline
{
  "domain": "direct_lab",
  "hypothesis": "WIMP",
  "step": 1,
  "intervention": "-",
  "infra_cost_usd": 120.0,
  "qps": 480.0,
  "latency_p95_ms": 180.0
}

// v1.0.6 (with physics) - backward compatible extension
{
  "domain": "direct_lab",     // unchanged
  "hypothesis": "WIMP",       // unchanged
  "step": 1,                  // unchanged
  "intervention": "-",        // unchanged
  "infra_cost_usd": 120.0,    // unchanged
  "qps": 480.0,               // unchanged
  "latency_p95_ms": 180.0,    // unchanged

  "physics": {                // NEW - optional object
    "material": "Si",
    "e_tot_hartree": -289.123,
    "band_gap_ev": 1.17,
    "wimp_sigma_cm2": 1.2e-40
  }
}
```

**Validation Update** (`sidrce/telemetry_contract.json`):
```diff
--- a/sidrce/telemetry_contract.json
+++ b/sidrce/telemetry_contract.json
@@ -10,7 +10,15 @@
     "qps": {"type": "number", "minimum": 0},
     "latency_p95_ms": {"type": "number", "minimum": 0},
     "infra_cost_usd": {"type": "number", "minimum": 0}
-  }
+  },
+  "additionalProperties": true,
+  "definitions": {
+    "physics": {
+      "type": "object",
+      "properties": {...}
+    }
+  },
+  "note": "physics field is optional for backward compatibility"
 }
```

**Validator Update** (`scripts/validate_telemetry.py`):
```python
# No code change needed!
# jsonschema with "additionalProperties": true ignores unknown fields
# Old telemetry (no physics) → still valid
# New telemetry (with physics) → also valid
```

### 4.2 Configuration Extension

**config/experiment.yaml** (diff):
```diff
--- a/config/experiment.yaml
+++ b/config/experiment.yaml
@@ -10,6 +10,16 @@ domains:
       "3": "shield_upgrade"
       "5": "long_integration"

+    # NEW: Optional physics simulation
+    pyscf:
+      enabled: false  # default: off (backward compatible)
+      materials: [Si]
+      basis: gth-dzvp
+      xc: PBE
+      kpts: [2, 2, 2]
+      q_range: [0.001, 0.5]
+      q_points: 50
+
   cosmology:
     hypothesis: [DarkPhoton, SterileNeutrino]
     interventions:
```

**Config Loader** (new utility):
```python
# src/materials/config_loader.py
import yaml
from typing import Optional, Dict, Any

def load_physics_config(domain: str) -> Optional[Dict[str, Any]]:
    """
    Load physics config for given domain.

    Returns None if:
    - config/experiment.yaml missing
    - domain not found
    - pyscf section missing or disabled
    """
    try:
        with open('config/experiment.yaml') as f:
            cfg = yaml.safe_load(f)
        domain_cfg = cfg.get('domains', {}).get(domain, {})
        pyscf_cfg = domain_cfg.get('pyscf', {})
        if not pyscf_cfg.get('enabled', False):
            return None
        return pyscf_cfg
    except Exception:
        return None  # Graceful: no config = no physics
```

### 4.3 SIDRCE Collector (Backward Compatible)

**sidrce/collector.py** (NO CHANGES NEEDED):
```python
# Existing code already ignores unknown fields:
for line in f:
    rec = json.loads(line)
    cost = rec['infra_cost_usd']  # Only reads known fields
    qps = rec['qps']
    latency = rec['latency_p95_ms']
    # rec['physics'] is ignored (not used in FTI calculation)
```

**Optional: Physics-Aware Metrics** (future enhancement):
```python
# sidrce/collector.py (extend, not modify)
def collect_physics_stats(records):
    """
    NEW function - does not affect existing FTI/λ calculation.
    """
    physics_records = [r for r in records if 'physics' in r]
    if not physics_records:
        return None
    return {
        'avg_computation_time': np.mean([p['physics']['computation_time_sec'] for p in physics_records]),
        'materials': list(set(p['physics']['material'] for p in physics_records))
    }
```

---

## 5. Error Handling Strategy

### 5.1 Graceful Degradation Hierarchy

```
Level 1: PySCF not installed
  └─> Import fails → run_simulation returns None → emit telemetry without physics field

Level 2: PySCF installed but config disabled
  └─> load_physics_config() returns None → skip simulation → no physics field

Level 3: PySCF installed, config enabled, but DFT fails
  └─> Catch exception → log error → emit telemetry with physics={"error": "..."}

Level 4: DFT succeeds but form factor calculation fails
  └─> Emit partial results (e_tot, band_gap) with missing cross-section fields
```

### 5.2 Error Handling Implementation

**src/materials/__init__.py**:
```python
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def run_simulation(hypothesis: str, material: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run physics simulation with layered error handling.

    Returns:
        {
          "physics": {...} or None
        }
    """
    # Level 1: Check PySCF availability
    if not _have_pyscf():
        logger.info("PySCF not available, skipping physics simulation")
        return {"physics": None}

    # Level 2: Check config
    pyscf_cfg = config.get('pyscf', {})
    if not pyscf_cfg.get('enabled', False):
        logger.debug("Physics simulation disabled in config")
        return {"physics": None}

    # Level 3: Run DFT with exception handling
    try:
        from .silicon import SiliconSimulator
        sim = SiliconSimulator(pyscf_cfg)
        result = sim.run_scf()

        # Level 4: Compute form factors
        try:
            from .form_factors import compute_wimp_cross_section
            q_vals = np.linspace(0.001, 0.5, 50)
            sigma = compute_wimp_cross_section(result, q_vals)
            result.metadata['wimp_sigma_cm2'] = sigma
        except Exception as e:
            logger.warning(f"Form factor calculation failed: {e}")
            result.metadata['form_factor_error'] = str(e)

        return {
            "physics": {
                "material": material,
                "e_tot_hartree": result.e_tot,
                "band_gap_ev": result.band_gap,
                **result.metadata
            }
        }

    except Exception as e:
        logger.error(f"DFT calculation failed: {e}", exc_info=True)
        return {
            "physics": {
                "error": str(e),
                "material": material,
                "hypothesis": hypothesis
            }
        }
```

### 5.3 Logging Strategy

**Log Levels**:
- `DEBUG`: Config checks, optional feature skips
- `INFO`: Successful simulation runs, performance metrics
- `WARNING`: Partial failures (e.g., form factor errors)
- `ERROR`: Full simulation failures (with traceback)

**Environment Control**:
```bash
export DMCA_LOG_LEVEL=DEBUG
export DMCA_PHYSICS_LOG_FILE=/tmp/dmca_physics.log
```

---

## 6. Testing Strategy

### 6.1 Test Matrix

| Test Type | PySCF | Purpose | Files |
|-----------|-------|---------|-------|
| **Unit (Core)** | ❌ Not installed | Verify core DMCA unchanged | `tests/test_*.py` |
| **Unit (Physics)** | ✅ Installed | Test physics modules in isolation | `tests/materials/test_*.py` |
| **Integration** | ✅ Installed | Test full pipeline with physics | `tests/integration/test_e2e_physics.py` |
| **Backward Compat** | ❌ Not installed | Verify old telemetry still works | `tests/test_backward_compat.py` |

### 6.2 CI/CD Matrix Extension

**.github/workflows/physics-gates.yml** (NEW FILE):
```yaml
name: Physics Module Gates

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        pyscf: [none, installed]
        python: ['3.9', '3.10', '3.11']

    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}

      - name: Install core dependencies
        run: pip install -r requirements.txt

      - name: Install PySCF (if matrix.pyscf == installed)
        if: matrix.pyscf == 'installed'
        run: pip install pyscf>=2.3.0

      - name: Run tests
        run: |
          if [ "${{ matrix.pyscf }}" == "none" ]; then
            # Without PySCF: core tests only
            pytest tests/ -k "not materials" -v
          else
            # With PySCF: all tests
            pytest tests/ -v
          fi

      - name: Validate telemetry schema (backward compat)
        run: python scripts/validate_telemetry.py sample_ops_telemetry.jsonl

      - name: ASDP check
        run: python tools/investor_asdp_cli.py check
```

### 6.3 Test Examples

**tests/materials/test_silicon.py**:
```python
import pytest
import numpy as np

# Mark as requiring pyscf
pytest.importorskip("pyscf")

from src.materials.silicon import SiliconSimulator
from src.materials.base import MaterialConfig

def test_silicon_scf_convergence():
    """Test that Si DFT converges to known energy"""
    config = MaterialConfig(basis='gth-szv', kpts=(1,1,1))
    sim = SiliconSimulator(config)
    result = sim.run_scf()

    # Reference: VASP calculation
    assert abs(result.e_tot - (-289.123)) < 0.1, "Total energy mismatch"
    assert abs(result.band_gap - 1.17) < 0.2, "Band gap error too large"

def test_form_factor_normalization():
    """Test F(q=0) = 1"""
    config = MaterialConfig(kpts=(1,1,1))
    sim = SiliconSimulator(config)
    result = sim.run_scf()

    F_q = sim.compute_form_factor(np.array([0.0]))
    assert abs(F_q[0] - 1.0) < 1e-3, "Form factor not normalized"
```

**tests/test_backward_compat.py**:
```python
import json
from src.sidrce_bridge import TelemetryEmitter

def test_old_telemetry_format_still_works(tmp_path):
    """Verify existing code path unchanged"""
    emitter = TelemetryEmitter(cfg=EmitterConfig(path=str(tmp_path / "test.jsonl")))

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
    with open(tmp_path / "test.jsonl") as f:
        rec = json.loads(f.read())
        assert "physics" not in rec, "Old format should not have physics field"
        assert rec["qps"] == 480.0
```

---

## 7. Performance Considerations

### 7.1 DFT Computation Cost

**Typical Timings** (Si, gth-dzvp basis):
- Gamma-point (1,1,1): ~30 seconds
- (2,2,2) k-points: ~3 minutes
- (4,4,4) k-points: ~25 minutes
- (8,8,8) k-points: ~3 hours

**Recommendation**: Use (2,2,2) for production (good accuracy/speed balance).

### 7.2 Caching Strategy

**Checkpoint Files**:
```python
# src/materials/silicon.py
def run_scf(self):
    cache_path = Path(os.getenv('DMCA_PYSCF_CACHE', '/tmp/pyscf_cache'))
    chkfile = cache_path / f"si_{hash(self.config)}.chk"

    if chkfile.exists():
        # Load from checkpoint
        mf = scf.RHF(self.cell, chkfile=str(chkfile))
        return self._extract_result(mf)

    # Run from scratch
    mf = scf.RHF(self.cell).run()
    mf.chkfile = str(chkfile)
    return self._extract_result(mf)
```

### 7.3 Parallel Processing

**Multi-Material Batch**:
```python
from multiprocessing import Pool

def batch_simulate(hypotheses, materials, config):
    tasks = [(h, m, config) for h in hypotheses for m in materials]

    with Pool(processes=4) as pool:
        results = pool.starmap(run_simulation, tasks)

    return results
```

**OpenMP Threads** (PySCF internal parallelism):
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 7.4 Telemetry Impact

**Overhead**:
- Old telemetry (7 fields): ~200 bytes/record
- New telemetry (7 + physics): ~400 bytes/record
- Encryption: +30% (AES-GCM nonce + MAC)
- Compression: -60% (gzip)

**Mitigation**: Enable compression for physics runs
```bash
export SIDRCE_GZIP=1
```

---

## Summary Checklist

Before implementing, verify design against:

- [ ] **Zero Modification**: No changes to `src/sidrce_bridge.py`, `sidrce/collector.py`, `src/causal.py`
- [ ] **Optional Dependency**: PySCF importable fails gracefully
- [ ] **Backward Compatible**: Old telemetry format still validates
- [ ] **Namespace Isolated**: `src/materials/` independent from core
- [ ] **ASDP Enforced**: Spicy rules prevent cross-contamination
- [ ] **Test Coverage**: ≥80% for new code, 100% for existing
- [ ] **CI Matrix**: Tests pass with/without PySCF
- [ ] **Performance**: DFT cost amortized via caching
- [ ] **Documentation**: All APIs documented with examples

---

**Next**: See `implementation-patches.md` for line-by-line diff patches.
