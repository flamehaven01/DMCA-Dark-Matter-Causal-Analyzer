# Quality Assurance Checklist
## Zero-Drift Validation for Physics Module Integration

**Document Version**: 1.0
**Date**: 2025-11-10
**Purpose**: Ensure 0% code drift from DMCA v1.0.5 core functionality

---

## Table of Contents

1. [Pre-Implementation Checklist](#1-pre-implementation-checklist)
2. [During Implementation](#2-during-implementation)
3. [Post-Implementation Validation](#3-post-implementation-validation)
4. [Performance Benchmarks](#4-performance-benchmarks)
5. [Security Audit](#5-security-audit)
6. [Documentation Review](#6-documentation-review)
7. [Release Criteria](#7-release-criteria)

---

## 1. Pre-Implementation Checklist

### 1.1 Baseline Capture

**Before applying ANY patches**, capture baseline metrics:

```bash
# Checkpoint current state
git checkout -b baseline-v1.0.5
git tag v1.0.5-baseline

# Run all existing tests
pytest tests/ -v --cov=src --cov-report=html --cov-report=term > baseline_tests.log

# ASDP score
python tools/investor_asdp_cli.py check > baseline_asdp.json

# Telemetry validation
python scripts/validate_telemetry.py sample_ops_telemetry.jsonl

# SIDRCE collector
python sidrce/collector.py > baseline_sidrce.json

# Record file checksums
find src/ sidrce/ -name "*.py" -exec md5sum {} \; > baseline_checksums.txt
```

**Save These Files** (for comparison after integration):
- `baseline_tests.log`
- `baseline_asdp.json`
- `baseline_sidrce.json`
- `baseline_checksums.txt`

### 1.2 Environment Setup

```bash
# Create clean virtual environment
python3 -m venv venv_physics_test
source venv_physics_test/bin/activate

# Install only core dependencies (NO PySCF)
pip install -r requirements.txt
pip install pytest pytest-cov

# Verify PySCF NOT installed
python -c "import pyscf" 2>&1 | grep -q "ModuleNotFoundError" && echo "OK: PySCF not found"
```

---

## 2. During Implementation

### 2.1 After Each Patch Application

**Run this checklist after EVERY patch (1-10)**:

```bash
# 1. Syntax check
python -m py_compile src/materials/*.py

# 2. Import test (without PySCF)
python -c "from src.sidrce_bridge import TelemetryEmitter; print('Core intact')"

# 3. Run core tests (should still pass)
pytest tests/ -k "not materials" -v

# 4. Check file modifications
md5sum src/sidrce_bridge.py src/causal.py sidrce/collector.py sidrce/server.py
# Compare against baseline_checksums.txt
# These files should NEVER change (md5sum must match exactly)
```

**Critical**: If ANY core file checksum changes, STOP immediately and revert the patch.

### 2.2 Incremental Integration Test

After Patch 5 (physics_emitter.py), verify wrapper pattern:

```bash
cat > test_integration_minimal.py <<'EOF'
from src.materials.physics_emitter import PhysicsEmitter
import tempfile
from pathlib import Path

# Test 1: PhysicsEmitter creates base emitter automatically
emitter = PhysicsEmitter()
assert emitter.base_emitter is not None, "Auto-creation failed"

# Test 2: Emit without physics (backward compat)
with tempfile.TemporaryDirectory() as tmp:
    from src.sidrce_bridge import TelemetryEmitter, EmitterConfig
    base = TelemetryEmitter(cfg=EmitterConfig(path=f"{tmp}/test.jsonl", batch_size=1))
    emitter = PhysicsEmitter(base_emitter=base)

    emitter.emit_physics(
        domain="direct_lab",
        hypothesis="WIMP",
        step=1,
        intervention="-",
        infra_cost_usd=120.0,
        qps=480.0,
        latency_p95_ms=180.0,
        physics_result=None
    )
    emitter.flush()

    import json
    with open(f"{tmp}/test.jsonl") as f:
        rec = json.loads(f.read())
    assert "physics" not in rec, "Should not add physics when None"
    print("✓ Backward compatibility intact")

print("✓ All integration tests passed")
EOF

python test_integration_minimal.py
```

---

## 3. Post-Implementation Validation

### 3.1 Core Integrity Check (CRITICAL)

**These files MUST be byte-identical to baseline:**

```bash
# Compare checksums
md5sum src/sidrce_bridge.py | diff - <(grep sidrce_bridge.py baseline_checksums.txt)
md5sum src/causal.py | diff - <(grep causal.py baseline_checksums.txt)
md5sum sidrce/collector.py | diff - <(grep collector.py baseline_checksums.txt)
md5sum sidrce/server.py | diff - <(grep server.py baseline_checksums.txt)

# If ANY diff output appears, FAIL validation immediately
```

**Expected Output**: (empty - no differences)

### 3.2 Backward Compatibility Test Suite

**Test Matrix**:

| Test | PySCF | Config | Expected Behavior |
|------|-------|--------|-------------------|
| 1 | ❌ | N/A | All core tests pass |
| 2 | ❌ | N/A | Old telemetry validates |
| 3 | ❌ | N/A | SIDRCE collector runs |
| 4 | ✅ | disabled | Core tests pass |
| 5 | ✅ | enabled | Core + physics tests pass |

**Run Tests**:

```bash
# Test 1-3: Without PySCF
pip uninstall -y pyscf
pytest tests/ -k "not materials" -v --tb=short
python scripts/validate_telemetry.py sample_ops_telemetry.jsonl
python sidrce/collector.py

# Test 4: With PySCF but config disabled
pip install pyscf>=2.3.0
export DMCA_PHYSICS_ENABLED=0
pytest tests/ -k "not materials" -v

# Test 5: With PySCF and config enabled
export DMCA_PHYSICS_ENABLED=1
pytest tests/materials/ -v -m "not slow"
```

**All tests must pass (100% success rate).**

### 3.3 Schema Validation

**Test backward compatibility of telemetry schema:**

```bash
# Create test telemetry files
cat > old_format.jsonl <<EOF
{"domain":"direct_lab","hypothesis":"WIMP","step":1,"intervention":"-","infra_cost_usd":120.0,"qps":480.0,"latency_p95_ms":180.0}
EOF

cat > new_format.jsonl <<EOF
{"domain":"direct_lab","hypothesis":"WIMP","step":1,"intervention":"-","infra_cost_usd":120.0,"qps":480.0,"latency_p95_ms":180.0,"physics":{"material":"Si","e_tot_hartree":-289.123,"band_gap_ev":1.17}}
EOF

# Both should validate
python scripts/validate_telemetry.py old_format.jsonl
echo $?  # Must be 0

python scripts/validate_telemetry.py new_format.jsonl
echo $?  # Must be 0

# SIDRCE collector should ingest both
python sidrce/collector.py old_format.jsonl
python sidrce/collector.py new_format.jsonl
```

**Expected**: No errors, FTI calculations identical (physics field ignored by collector).

### 3.4 ASDP Quality Gate

```bash
# Run ASDP check
python tools/investor_asdp_cli.py check > post_integration_asdp.json

# Compare Ω score
BASELINE_OMEGA=$(grep '"omega":' baseline_asdp.json | grep -oP '[\d.]+')
CURRENT_OMEGA=$(grep '"omega":' post_integration_asdp.json | grep -oP '[\d.]+')

if (( $(echo "$CURRENT_OMEGA < $BASELINE_OMEGA" | bc -l) )); then
    echo "FAIL: Ω score degraded ($BASELINE_OMEGA → $CURRENT_OMEGA)"
    exit 1
else
    echo "PASS: Ω score maintained or improved ($BASELINE_OMEGA → $CURRENT_OMEGA)"
fi

# Check for new SAST issues
grep '"high":' post_integration_asdp.json | grep -v '"high": 0' && {
    echo "FAIL: New high-severity SAST issues found"
    exit 1
}

# Check for secrets
grep '"secrets":' post_integration_asdp.json | grep -v '"secrets": 0' && {
    echo "FAIL: Secrets detected in new code"
    exit 1
}

echo "✓ ASDP quality gate passed"
```

### 3.5 CI/CD Gate Validation

```bash
# Simulate GitHub Actions locally
act -j test-physics -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest

# Or run matrix manually
for pyscf in none installed; do
    for python_ver in 3.9 3.10 3.11; do
        echo "Testing: Python $python_ver, PySCF $pyscf"

        # Create fresh venv
        python${python_ver} -m venv venv_test_${python_ver}_${pyscf}
        source venv_test_${python_ver}_${pyscf}/bin/activate

        pip install -r requirements.txt pytest

        if [ "$pyscf" = "installed" ]; then
            pip install pyscf>=2.3.0
        fi

        # Run appropriate tests
        if [ "$pyscf" = "none" ]; then
            pytest tests/ -k "not materials" -v
        else
            pytest tests/ -v
        fi

        deactivate
    done
done
```

**Expected**: All 6 combinations (3 Python × 2 PySCF) pass.

---

## 4. Performance Benchmarks

### 4.1 Telemetry Overhead

**Measure emission latency:**

```python
import time
from src.sidrce_bridge import TelemetryEmitter, EmitterConfig
from src.materials.physics_emitter import PhysicsEmitter

# Baseline: Original emitter
emitter_base = TelemetryEmitter(cfg=EmitterConfig(path="/tmp/bench_base.jsonl"))
payload = {
    "domain": "direct_lab",
    "hypothesis": "WIMP",
    "step": 1,
    "intervention": "-",
    "infra_cost_usd": 120.0,
    "qps": 480.0,
    "latency_p95_ms": 180.0
}

t0 = time.perf_counter()
for _ in range(1000):
    emitter_base.emit(payload)
emitter_base.flush()
t_base = time.perf_counter() - t0

# Physics emitter (no physics)
emitter_phys = PhysicsEmitter(base_emitter=TelemetryEmitter(
    cfg=EmitterConfig(path="/tmp/bench_phys.jsonl")
))

t0 = time.perf_counter()
for _ in range(1000):
    emitter_phys.emit_physics(physics_result=None, **payload)
emitter_phys.flush()
t_phys = time.perf_counter() - t0

overhead = (t_phys - t_base) / t_base * 100
print(f"Base: {t_base:.3f}s, Physics: {t_phys:.3f}s, Overhead: {overhead:.1f}%")

# PASS criteria: Overhead < 10%
assert overhead < 10, f"Excessive overhead: {overhead:.1f}%"
```

**Target**: Overhead < 10% (acceptable for wrapper pattern).

### 4.2 SIDRCE Collector Performance

```bash
# Generate large telemetry file (10k records)
python -c "
import json
for i in range(10000):
    print(json.dumps({
        'domain': 'direct_lab',
        'hypothesis': 'WIMP',
        'step': i,
        'intervention': '-',
        'infra_cost_usd': 120.0,
        'qps': 480.0,
        'latency_p95_ms': 180.0
    }))
" > large_telemetry.jsonl

# Benchmark collector
time python sidrce/collector.py large_telemetry.jsonl > /dev/null
# Should complete in < 1 second
```

### 4.3 DFT Computation Benchmarks

**Baseline timings** (for future optimization):

```bash
pip install pyscf>=2.3.0

python <<EOF
import time
from src.materials.silicon import SiliconSimulator
from src.materials.base import MaterialConfig

for kpts in [(1,1,1), (2,2,2)]:
    config = MaterialConfig(basis='gth-szv', kpts=kpts)
    sim = SiliconSimulator(config)

    t0 = time.time()
    result = sim.run_scf()
    elapsed = time.time() - t0

    print(f"kpts={kpts}: {elapsed:.1f}s, Eg={result.band_gap:.3f} eV")
EOF
```

**Expected**:
- (1,1,1): 10-30 seconds
- (2,2,2): 1-3 minutes

---

## 5. Security Audit

### 5.1 Dependency Check

```bash
# Install safety
pip install safety

# Check for known vulnerabilities
safety check --json > safety_report.json

# FAIL if any high-severity issues found
grep '"severity": "high"' safety_report.json && {
    echo "FAIL: High-severity vulnerabilities found"
    exit 1
}
```

### 5.2 Secrets Scan

```bash
# Use ASDP or additional tools
python tools/investor_asdp_cli.py check | grep -i secret

# Or use gitleaks
gitleaks detect --source . --report-path gitleaks_report.json
```

### 5.3 Code Review Checklist

Manual review of new code:

- [ ] No hardcoded credentials or API keys
- [ ] No SQL injection vectors (N/A - no database)
- [ ] No command injection (check `subprocess` calls)
- [ ] No XSS vectors (N/A - server-side only)
- [ ] Proper input validation in `run_simulation()`
- [ ] Exception handling doesn't leak sensitive info
- [ ] File I/O uses safe paths (no `../` traversal)

**Example**: Check physics modules for unsafe patterns:

```bash
# Search for dangerous patterns
grep -r "subprocess\|eval\|exec\|os.system" src/materials/
# Should return empty

grep -r "__import__" src/materials/
# Should return empty (dynamic imports risky)
```

---

## 6. Documentation Review

### 6.1 Completeness Check

Verify all documents exist and are complete:

```bash
ls -lh docs/
# Should contain:
# - pyscf-integration-technical-guide.md
# - physics-simulation-mathematics.md
# - architecture-integration-design.md
# - implementation-patches.md
# - quality-assurance-checklist.md (this file)

# Check word counts (ensure sufficient detail)
wc -w docs/*.md
# Each should be > 2000 words
```

### 6.2 Internal Consistency

Cross-reference between documents:

- [ ] All API examples in `pyscf-integration-technical-guide.md` match code in `implementation-patches.md`
- [ ] Physics equations in `physics-simulation-mathematics.md` match implementation in `form_factors.py`
- [ ] Architecture diagrams in `architecture-integration-design.md` match module structure
- [ ] Patch sequence in `implementation-patches.md` follows roadmap in `pyscf-integration-technical-guide.md`

### 6.3 Code Examples Validation

Extract and test all code examples:

```bash
# Extract Python code blocks from docs
for doc in docs/*.md; do
    echo "Testing code examples in $doc"
    # Use a markdown-code-extractor tool or manual extraction
done
```

---

## 7. Release Criteria

### 7.1 Go/No-Go Checklist

**CRITICAL** - All items must be ✅ to release:

#### Core Integrity (0% Drift)
- [ ] src/sidrce_bridge.py md5sum matches baseline
- [ ] src/causal.py md5sum matches baseline
- [ ] sidrce/collector.py md5sum matches baseline
- [ ] sidrce/server.py md5sum matches baseline

#### Backward Compatibility
- [ ] All v1.0.5 tests pass without PySCF
- [ ] Old telemetry format validates
- [ ] SIDRCE collector processes old format
- [ ] No API breaking changes

#### Testing
- [ ] Unit test coverage ≥ 80% for new code
- [ ] All CI/CD matrix combinations pass (6/6)
- [ ] Physics tests pass with PySCF installed
- [ ] Backward compat tests pass

#### Quality
- [ ] ASDP Ω score ≥ baseline (≥ 0.8)
- [ ] Zero high-severity SAST issues
- [ ] Zero secrets detected
- [ ] Zero high-severity vulnerabilities (safety check)

#### Performance
- [ ] PhysicsEmitter overhead < 10%
- [ ] SIDRCE collector performance unchanged
- [ ] DFT computations within expected range

#### Documentation
- [ ] All 5 doc files complete (> 2000 words each)
- [ ] Code examples tested and working
- [ ] Internal cross-references validated
- [ ] README.md updated with v1.0.6 features

#### Integration
- [ ] config/experiment.yaml extended (backward compatible)
- [ ] sidrce/telemetry_contract.json extended (additionalProperties: true)
- [ ] ASDP rules added (no cross-contamination)
- [ ] CI/CD workflow added (physics-gates.yml)

### 7.2 Release Procedure

If all criteria met:

```bash
# 1. Final ASDP check
python tools/investor_asdp_cli.py auto

# 2. Version bump
echo "1.0.6" > VERSION

# 3. Update CHANGELOG
cat >> CHANGELOG.md <<EOF
## [1.0.6] - $(date +%Y-%m-%d)
### Added
- PySCF-based quantum chemistry module for dark matter simulations
- Silicon (Si) and Germanium (Ge) DFT calculations
- WIMP, Axion, DarkPhoton, SterileNeutrino form factors
- PhysicsEmitter wrapper for telemetry extension
- Comprehensive physics documentation (5 guides)

### Changed
- Extended telemetry schema with optional 'physics' field
- Enhanced config/experiment.yaml with pyscf section
- Added physics-gates.yml CI/CD workflow
- Updated ASDP rules for module isolation

### Fixed
- N/A (new feature, no bugs fixed)

### Backward Compatibility
- 100% compatible with v1.0.5 telemetry format
- All core modules unchanged (0% drift)
- PySCF is optional dependency
EOF

# 4. Git commit
git add .
git commit -m "feat: Add PySCF quantum chemistry module (v1.0.6)

- Zero-drift integration (core modules unchanged)
- Optional PySCF dependency with graceful degradation
- Comprehensive physics simulation and documentation
- ASDP Ω score: $(grep omega post_integration_asdp.json | grep -oP '[\d.]+')
- Test coverage: 100% backward compat, 85% new code
"

# 5. Tag release
git tag -a v1.0.6 -m "DMCA v1.0.6 - Physics Module"

# 6. Push to branch
git push -u origin claude/dmca-v105-finops-separation-011CUzUnvppYbgYLHew42kkT
git push --tags
```

---

## 8. Post-Release Monitoring

### 8.1 Smoke Test in Production

After merge to main:

```bash
# Pull latest
git pull origin main

# Full test suite
pytest tests/ -v --cov=src --cov-report=html

# Generate physics telemetry (if PySCF available)
python <<EOF
from src.materials import run_simulation
from src.materials.physics_emitter import PhysicsEmitter

result = run_simulation('WIMP', 'Si', {'kpts': (1,1,1)})
print("Physics simulation result:", result)

emitter = PhysicsEmitter()
emitter.emit_physics(
    domain='direct_lab',
    hypothesis='WIMP',
    step=1,
    intervention='-',
    infra_cost_usd=120.0,
    qps=480.0,
    latency_p95_ms=180.0,
    physics_result=result
)
emitter.flush()
print("✓ Telemetry emitted successfully")
EOF

# Verify SIDRCE ingestion
python sidrce/collector.py ops_telemetry.jsonl
```

### 8.2 User Acceptance Criteria

Coordinate with team receiving code:

- [ ] Documentation is clear and complete
- [ ] Code examples work as-is (copy-paste)
- [ ] Installation instructions accurate
- [ ] Troubleshooting section covers common issues
- [ ] Performance is acceptable for use case

---

## Summary

**Zero-Drift Achieved If**:
- All 4 core files unchanged (byte-identical)
- All backward compatibility tests pass (100%)
- ASDP Ω score ≥ baseline
- No new SAST/security issues
- Documentation complete and validated

**Report Template**:

```
DMCA v1.0.6 Physics Module Integration - QA Report
==================================================
Date: $(date)
Tester: [Name]

CORE INTEGRITY: [✅/❌]
  - sidrce_bridge.py: [✅/❌]
  - causal.py: [✅/❌]
  - collector.py: [✅/❌]
  - server.py: [✅/❌]

BACKWARD COMPATIBILITY: [✅/❌]
  - Tests without PySCF: [PASS/FAIL]
  - Old telemetry validation: [PASS/FAIL]
  - SIDRCE collector: [PASS/FAIL]

QUALITY: [✅/❌]
  - ASDP Ω: [score] (baseline: [baseline])
  - SAST High: [count]
  - Secrets: [count]
  - Vulnerabilities: [count]

PERFORMANCE: [✅/❌]
  - Telemetry overhead: [%]
  - Collector time: [seconds]

TESTING: [✅/❌]
  - Unit tests: [pass/total]
  - Integration tests: [pass/total]
  - CI/CD matrix: [pass/total]

DOCUMENTATION: [✅/❌]
  - Completeness: [✅/❌]
  - Code examples: [✅/❌]
  - Internal consistency: [✅/❌]

OVERALL: [✅ GO / ❌ NO-GO]
```

---

**End of Quality Assurance Checklist**
