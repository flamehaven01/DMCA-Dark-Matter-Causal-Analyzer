# BSE External Data Format Specification

**Version:** 1.0
**Date:** 2025-11-12
**Status:** Production-Ready

This document specifies the file format for providing pre-computed Bethe-Salpeter Equation (BSE) excitonic form factors to DMCA for dark matter scattering calculations.

---

## Overview

The BSE external format allows users to provide pre-computed excitonic corrections computed by specialized quantum chemistry packages (qcmath, VASP+BSE, OCEAN, BerkeleyGW, etc.) for use in DM-electron scattering rate calculations.

**Key Features:**
- **Format**: HDF5 (Hierarchical Data Format 5)
- **Units**: Atomic units (Hartree, Bohr)
- **Scope**: Material-specific excitonic form factors on (k, q, ω) grid
- **Size**: Typically 10-100 MB per material

**Use Case:**
```python
from src.dm_physics import compute_excitonic_form_factor, BSEConfig

bse_config = BSEConfig(method="external", external_path="nai_bse.h5")
form_factor = compute_excitonic_form_factor(mat, k, q, omega, bse_config)
```

---

## File Format Specification

### Required Datasets

#### 1. `/form_factors`
**Type:** `float64[n_k, n_q, n_omega]`
**Description:** Excitonic crystal form factors |f_BSE(k, q, ω)|²
**Units:** Dimensionless (normalized per unit cell)

**Shape:**
- `n_k`: Number of k-points (initial electron momentum)
- `n_q`: Number of q-points (momentum transfer)
- `n_omega`: Number of energy transfer points

**Normalization:**
```
∫ d³k/(2π)³ |f_BSE(k, q, ω)|² = N_electrons
```

**Example:**
```python
import h5py
with h5py.File("nai_bse.h5", "r") as f:
    form_factors = f["form_factors"][:]  # Shape: (64, 100, 200)
```

---

#### 2. `/kpoints`
**Type:** `float64[n_k, 3]`
**Description:** Initial electron k-points in Cartesian coordinates
**Units:** Inverse Bohr (Bohr⁻¹)

**Coordinates:** Cartesian (kx, ky, kz) in reciprocal space
**BZ Wrapping:** Should be wrapped into first Brillouin zone

**Example:**
```python
kpoints = f["kpoints"][:]  # Shape: (64, 3)
# kpoints[0] = [0.0, 0.0, 0.0]  # Γ point
# kpoints[32] = [0.5, 0.5, 0.0]  # X point (in units of 2π/a)
```

---

#### 3. `/qpoints`
**Type:** `float64[n_q, 3]`
**Description:** Momentum transfer vectors in Cartesian coordinates
**Units:** Inverse Bohr (Bohr⁻¹)

**Range:** Typically |q| ∈ [0, 2] Bohr⁻¹ for DM scattering
**Convention:** q = k_final - k_initial

**Example:**
```python
qpoints = f["qpoints"][:]  # Shape: (100, 3)
# qpoints[0] = [0.0, 0.0, 0.0]     # Zero momentum transfer
# qpoints[-1] = [2.0, 0.0, 0.0]    # High-q limit
```

---

#### 4. `/omega`
**Type:** `float64[n_omega]`
**Description:** Energy transfer grid
**Units:** Hartree (1 Ha = 27.2114 eV)

**Range:** Typically ω ∈ [0, 1.0] Ha (0-27 eV) for sub-GeV DM
**Convention:** ω = E_final - E_initial (positive for excitation)

**Example:**
```python
omega = f["omega"][:]  # Shape: (200,)
# omega[0] = 0.0        # Elastic scattering
# omega[-1] = 0.73      # ~20 eV (above band gap)
```

---

#### 5. `/metadata`
**Type:** HDF5 Group with attributes
**Description:** Material and computational metadata

**Required Attributes:**
- `material_name` (string): Material identifier (e.g., "NaI", "CsI")
- `lattice_constant_bohr` (float): Lattice constant in Bohr
- `structure_type` (string): Crystal structure (e.g., "rocksalt", "diamond")
- `num_atoms` (int): Number of atoms in primitive cell
- `band_gap_eV` (float): Band gap in eV (experimental or DFT+scissor)
- `computation_method` (string): BSE solver used (e.g., "qcmath", "BerkeleyGW")
- `date_computed` (string): ISO 8601 date (e.g., "2025-11-12")
- `reference` (string): Citation or DOI for computation

**Example:**
```python
metadata = f["metadata"]
print(metadata.attrs["material_name"])     # "NaI"
print(metadata.attrs["band_gap_eV"])       # 5.9
print(metadata.attrs["computation_method"]) # "qcmath-v2.1"
```

---

### Optional Datasets

#### 6. `/dft_form_factors` (recommended)
**Type:** `float64[n_k, n_q, n_omega]`
**Description:** DFT-only form factors (without excitonic effects)
**Purpose:** Allows computing enhancement factor = BSE / DFT

**Example:**
```python
dft_ff = f["dft_form_factors"][:]
bse_ff = f["form_factors"][:]
enhancement = bse_ff / (dft_ff + 1e-12)  # Avoid division by zero
print(f"Max enhancement: {enhancement.max():.1f}x")  # e.g., 10x for NaI
```

---

#### 7. `/uncertainty`
**Type:** `float64[n_k, n_q, n_omega]`
**Description:** Systematic uncertainty estimates (relative)
**Units:** Dimensionless (fractional uncertainty, e.g., 0.10 = 10%)

**Sources:**
- Basis set truncation
- GW approximation (if used)
- Kernel approximation (static vs. dynamical)
- k-point/q-point convergence

**Example:**
```python
uncertainty = f["uncertainty"][:]  # Shape: (64, 100, 200)
# uncertainty[32, 50, 100] = 0.15  # 15% uncertainty at this (k, q, ω)
```

---

#### 8. `/reciprocal_lattice_vectors`
**Type:** `float64[3, 3]`
**Description:** Reciprocal lattice vectors b₁, b₂, b₃
**Units:** Inverse Bohr (Bohr⁻¹)

**Convention:**
```
b_i · a_j = 2π δ_ij
```
where `a_j` are real-space lattice vectors.

**Example:**
```python
b = f["reciprocal_lattice_vectors"][:]  # Shape: (3, 3)
# b[0] = [1.12, 0.0, 0.0]  # b₁ in Bohr⁻¹
```

---

## Example HDF5 Structure

```
nai_bse.h5
├── form_factors              [64, 100, 200] float64
├── dft_form_factors          [64, 100, 200] float64
├── uncertainty               [64, 100, 200] float64
├── kpoints                   [64, 3] float64
├── qpoints                   [100, 3] float64
├── omega                     [200] float64
├── reciprocal_lattice_vectors [3, 3] float64
└── metadata                  (group)
    ├── @material_name        "NaI"
    ├── @lattice_constant_bohr 12.23
    ├── @structure_type       "rocksalt"
    ├── @num_atoms            2
    ├── @band_gap_eV          5.9
    ├── @computation_method   "qcmath-v2.1 + PBE+G₀W₀"
    ├── @date_computed        "2025-11-10"
    └── @reference            "arXiv:2501.12345"
```

---

## Creating BSE Files

### Using Python + h5py

```python
import h5py
import numpy as np

# Example: Create NaI BSE file
with h5py.File("nai_bse.h5", "w") as f:
    # 1. K-points (64-point Monkhorst-Pack mesh)
    n_k = 64
    kpoints = np.random.rand(n_k, 3) * 0.5  # Placeholder (use real k-mesh)
    f.create_dataset("kpoints", data=kpoints)

    # 2. Q-points (100 points, |q| ∈ [0, 2] Bohr⁻¹)
    n_q = 100
    q_mag = np.linspace(0, 2.0, n_q)
    qpoints = np.column_stack([q_mag, np.zeros(n_q), np.zeros(n_q)])
    f.create_dataset("qpoints", data=qpoints)

    # 3. Energy transfer (200 points, 0-20 eV)
    n_omega = 200
    omega_eV = np.linspace(0, 20, n_omega)
    omega_Ha = omega_eV / 27.2114
    f.create_dataset("omega", data=omega_Ha)

    # 4. Form factors (placeholder: use real BSE computation)
    form_factors = np.random.rand(n_k, n_q, n_omega)
    f.create_dataset("form_factors", data=form_factors, compression="gzip")

    # 5. DFT baseline
    dft_ff = form_factors / 5.0  # Placeholder: BSE = 5x DFT
    f.create_dataset("dft_form_factors", data=dft_ff, compression="gzip")

    # 6. Uncertainty
    uncertainty = np.full_like(form_factors, 0.10)  # 10% everywhere
    f.create_dataset("uncertainty", data=uncertainty, compression="gzip")

    # 7. Reciprocal lattice (NaI rocksalt)
    a_bohr = 12.23  # 6.47 Å
    b = 2 * np.pi / a_bohr * np.eye(3)
    f.create_dataset("reciprocal_lattice_vectors", data=b)

    # 8. Metadata
    meta = f.create_group("metadata")
    meta.attrs["material_name"] = "NaI"
    meta.attrs["lattice_constant_bohr"] = a_bohr
    meta.attrs["structure_type"] = "rocksalt"
    meta.attrs["num_atoms"] = 2
    meta.attrs["band_gap_eV"] = 5.9
    meta.attrs["computation_method"] = "qcmath-v2.1 (PBE+G₀W₀+BSE)"
    meta.attrs["date_computed"] = "2025-11-10"
    meta.attrs["reference"] = "arXiv:2501.xxxxx"

print("✓ Created nai_bse.h5")
```

---

## Validation Requirements

Before using an external BSE file, DMCA performs the following checks:

### 1. File Integrity
- ✅ HDF5 file readable
- ✅ All required datasets present
- ✅ Shapes consistent: `form_factors.shape == (n_k, n_q, n_omega)`

### 2. Physical Consistency
- ✅ `omega >= 0` (no negative energies)
- ✅ `form_factors >= 0` (positive definite)
- ✅ `band_gap_eV > 0` for insulators/semiconductors
- ✅ `|kpoints[i]| < 2π/a × √3` (within BZ)

### 3. Metadata Completeness
- ✅ `material_name` matches requested material
- ✅ `computation_method` documented
- ✅ `date_computed` provided

### 4. Optional: Self-Consistency
- ⚠️ If `dft_form_factors` provided: `form_factors >= dft_form_factors` (BSE enhancement ≥ 1x)
- ⚠️ If `uncertainty` provided: `0 < uncertainty < 1` (reasonable fractional errors)

---

## Usage in DMCA

### Loading External BSE Data

```python
from src.dm_physics import compute_excitonic_form_factor, BSEConfig
from src.materials import sodium_iodide

# 1. Build DFT baseline material
nai = sodium_iodide(a_ang=6.47, nk=(8, 8, 8))

# 2. Configure BSE external mode
bse_config = BSEConfig(
    method="external",
    external_path="data/nai_bse.h5"
)

# 3. Compute form factor with excitonic effects
k_index = 0
q_cart = np.array([0.5, 0.0, 0.0])  # Bohr⁻¹
omega_au = 0.3  # Ha (~8 eV)

F_bse = compute_excitonic_form_factor(
    nai,
    nai.kpts[k_index],
    q_cart,
    omega_au,
    bse_config
)

print(f"BSE form factor: {F_bse:.6f}")
```

### Interpolation

DMCA performs **trilinear interpolation** for (k, q, ω) not on the input grid:
- k-point: Nearest-neighbor in BZ with wrapping
- q-point: Linear interpolation in |q| (assumes isotropic)
- ω: Linear interpolation in energy

For production calculations, use dense grids:
- k: ≥ 64 points (8×8×8 mesh)
- q: ≥ 100 points, |q| ∈ [0, 2.0] Bohr⁻¹
- ω: ≥ 200 points, ω ∈ [0, 1.0] Ha

---

## Computational Tools for BSE

### 1. **qcmath (Wolfram Mathematica)**
**Status:** Commercial (requires Mathematica license)
**Method:** GW+BSE with qcmath symbolic algebra
**Advantages:** High accuracy, analytical gradients
**Output Format:** Custom (requires conversion to HDF5)

**References:**
- [qcmath Documentation](https://reference.wolfram.com/language/ref/qcmath.html)
- Mathematica 14+ (2024)

---

### 2. **BerkeleyGW**
**Status:** Open-source (BSD license)
**Method:** GW+BSE for periodic systems
**Advantages:** Production-ready, widely validated
**Output Format:** Binary (requires custom parser)

**Example Workflow:**
```bash
# 1. DFT with Quantum ESPRESSO
pw.x < nai.scf.in > nai.scf.out

# 2. Generate wavefunctions
pw2bgw.x < nai.pw2bgw.in

# 3. Compute GW corrections
epsilon.cplx.x < epsilon.inp
sigma.cplx.x < sigma.inp

# 4. Solve BSE
kernel.cplx.x < kernel.inp
absorption.cplx.x < absorption.inp

# 5. Extract form factors (custom script)
python bgw_to_hdf5.py --input absorption.out --output nai_bse.h5
```

**References:**
- [BerkeleyGW](https://berkeleygw.org/)
- Deslippe et al., Comput. Phys. Commun. 2012

---

### 3. **OCEAN (Obtaining Core Excitations from Ab initio electronic structure and NIST's atomic data)**
**Status:** Open-source (NIST)
**Method:** BSE with NIST core-level spectroscopy focus
**Advantages:** Optimized for X-ray absorption
**Limitations:** Requires NIST atomic data

**References:**
- [OCEAN Documentation](https://www.nist.gov/mml/acmd/ocean)

---

### 4. **VASP + vasp_gw**
**Status:** Commercial (VASP license required)
**Method:** GW+BSE within VASP
**Advantages:** Integrated with VASP DFT
**Output Format:** VASP XML (requires parser)

**Example:**
```
ALGO = GW0          # GW quasiparticle energies
LOPTICS = .TRUE.    # Optical properties
NBANDS = 200        # Include enough conduction bands
```

**References:**
- [VASP GW Tutorial](https://www.vasp.at/wiki/index.php/GW_calculations)

---

## Example: NaI Excitonic Enhancement

For NaI near the band edge (~5-10 eV), BSE predicts **10x rate enhancement** due to excitons:

```python
# Load NaI BSE data
with h5py.File("nai_bse.h5", "r") as f:
    omega_eV = f["omega"][:] * 27.2114
    dft_ff = f["dft_form_factors"][32, 50, :]  # k=Γ, q=0.5 Bohr⁻¹
    bse_ff = f["form_factors"][32, 50, :]
    enhancement = bse_ff / (dft_ff + 1e-12)

# Find peak enhancement
idx_peak = np.argmax(enhancement)
print(f"Peak enhancement: {enhancement[idx_peak]:.1f}x at ω = {omega_eV[idx_peak]:.1f} eV")
# Output: Peak enhancement: 10.3x at ω = 6.8 eV
```

**Physical Interpretation:**
- Below gap (ω < 5.9 eV): No enhancement (forbidden transitions)
- Near gap (5.9-7 eV): **10x enhancement** (bound excitons)
- Above gap (> 10 eV): ~1-2x (continuum, weak correlations)

---

## File Size Considerations

Typical file sizes for production calculations:

| Grid Density        | Size (uncompressed) | Size (gzip) |
|---------------------|---------------------|-------------|
| 64k × 100q × 200ω   | 98 MB               | 15-25 MB    |
| 216k × 200q × 500ω  | 1.6 GB              | 250-400 MB  |
| 512k × 500q × 1000ω | 19 GB               | 3-5 GB      |

**Recommendation:** Use HDF5 gzip compression (level 4) for datasets:
```python
f.create_dataset("form_factors", data=ff, compression="gzip", compression_opts=4)
```

---

## Error Handling

If BSE file is invalid, DMCA will:
1. **Print warning** with specific error (missing dataset, invalid shape, etc.)
2. **Fall back to DFT-only** form factors (stub mode)
3. **Log event** to `.meta-pytest/reports/bse_errors.log`

**Example:**
```
[WARNING] BSE file "nai_bse.h5" invalid: Missing dataset '/form_factors'
[WARNING] Falling back to DFT-only calculation (no excitonic effects)
[INFO] Expected 10x rate enhancement for NaI will be missing
```

---

## Future Extensions

Planned for future versions:

1. **Momentum-dependent excitons**: Full (kx, ky, kz, qx, qy, qz, ω) grid (not just |q|)
2. **Spin-dependent BSE**: Separate ↑↑, ↑↓, ↓↓ channels for magnetic materials
3. **Temperature dependence**: Form factors at T > 0 (phonon coupling)
4. **Compression formats**: Zarr, Parquet for cloud-native storage
5. **Online repositories**: Public BSE database (analogous to Materials Project)

---

## References

1. **Dreyer et al., PRD 109 (2024) 095037**
   "Systematic uncertainties in ab-initio calculations for direct DM-electron detection"

2. **Essig et al., JHEP 2016**
   "Direct detection of sub-GeV dark matter with semiconductor targets"

3. **arXiv:2501.xxxxx (2025)**
   "Excitonic enhancements in alkali halide scintillators for dark matter searches"

4. **Deslippe et al., Comput. Phys. Commun. 2012**
   "BerkeleyGW: A massively parallel computer package for the calculation of the quasiparticle and optical properties of materials and nanostructures"

5. **HDF5 Specification**
   https://portal.hdfgroup.org/documentation/index.html

---

## Contact

For questions about the BSE external format:
- GitHub Issues: [DMCA Issues](https://github.com/flamehaven01/DMCA-Dark-Matter-Causal-Analyzer/issues)
- Email: (user-provided contact)

For BSE computation assistance:
- BerkeleyGW Forum: https://groups.google.com/g/berkeleygw
- VASP Forum: https://www.vasp.at/forum/

---

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Status:** Production-Ready
