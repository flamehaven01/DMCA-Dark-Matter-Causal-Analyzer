# Physics Simulation Mathematics
## Dark Matter Detection: Quantum Mechanics & Solid State Physics

**Document Version**: 1.0
**Date**: 2025-11-10
**Target Audience**: Theoretical physicists, quantum chemistry experts

---

## Table of Contents

1. [DFT Background](#1-dft-background)
2. [WIMP Direct Detection](#2-wimp-direct-detection)
3. [Axion Physics](#3-axion-physics)
4. [DarkPhoton Mixing](#4-darkphoton-mixing)
5. [SterileNeutrino Scattering](#5-sterileneutrino-scattering)
6. [Numerical Methods](#6-numerical-methods)
7. [References](#7-references)

---

## 1. DFT Background

### 1.1 Kohn-Sham Equations

The electronic structure is solved via density functional theory (DFT):

```
[-∇²/2 + V_ext(r) + V_H(r) + V_xc(r)] ψ_i(r) = ε_i ψ_i(r)
```

Where:
- `ψ_i(r)` = Kohn-Sham orbitals
- `ε_i` = Single-particle energies
- `V_ext(r)` = External potential (nuclei + lattice)
- `V_H(r)` = Hartree potential = ∫ n(r')/|r-r'| dr'
- `V_xc(r)` = Exchange-correlation potential (PBE, LDA, HSE, etc.)

**Electron Density**:
```
n(r) = Σ_i |ψ_i(r)|²
```

### 1.2 Periodic Boundary Conditions

For crystals, wavefunctions satisfy Bloch's theorem:
```
ψ_nk(r) = e^(ik·r) u_nk(r)
```
Where `u_nk(r)` has lattice periodicity.

**K-point Sampling**:
```
Observable = (1/N_k) Σ_k w_k f(k)
```
PySCF uses Monkhorst-Pack grids, e.g., `kpts=(2,2,2)` = 8 k-points.

### 1.3 Basis Sets

**Plane Wave Expansion** (PySCF default):
```
ψ_nk(r) = Σ_G c_nk(G) e^(i(k+G)·r)
```
Where `G` = reciprocal lattice vectors, truncated at `|k+G|² < E_cut`.

**GTH Pseudopotentials**:
- Replace core electrons with effective potential
- `gth-dzvp` = double-zeta valence + polarization
- Reduces computational cost by ~100x vs. all-electron

### 1.4 Band Structure

**Energy Bands**: `ε_n(k)` along high-symmetry path
```
Γ → X → W → L → Γ → K
```

**Band Gap**:
```
E_g = min_k[ε_c(k)] - max_k[ε_v(k)]
```
Where `c` = conduction band, `v` = valence band.

**Density of States**:
```
D(E) = Σ_nk δ(E - ε_nk)
```
Computed via tetrahedron method in PySCF.

---

## 2. WIMP Direct Detection

### 2.1 Scattering Cross-Section

**Spin-Independent (SI) Interaction**:
```
σ_SI = (4 μ²)/(π) × |F_p Z + F_n (A-Z)|² × |F_nuc(E_R)|²
```

Where:
- `μ = m_χ m_N / (m_χ + m_N)` = Reduced mass
- `F_p`, `F_n` = WIMP-proton/neutron couplings
- `F_nuc(E_R)` = **Nuclear form factor** (from DFT)

**Spin-Dependent (SD) Interaction**:
```
σ_SD = (32 G_F² μ²)/(π) × λ² J(J+1) × |F_nuc(E_R)|²
```
Where `λ = ⟨S_p⟩ a_p + ⟨S_n⟩ a_n` depends on nuclear spin structure.

### 2.2 Nuclear Form Factor (DFT Calculation)

**Definition**:
```
F_nuc(q) = (1/A) ∫ d³r ρ_nuc(r) e^(iq·r)
```

**DFT Implementation**:
```
ρ_nuc(r) = Σ_I Z_I δ(r - R_I) → Σ_I Z_I e^(iq·R_I)
```
For point nuclei. For finite-size nuclei:
```
ρ_nuc(r) = Σ_I Z_I ρ_Woods-Saxon(|r - R_I|)
```

**Helm Form Factor** (empirical):
```
F_Helm(q) = 3 [sin(qr_n) - qr_n cos(qr_n)] / (qr_n)³ × exp(-(qs)²/2)
```
Where `r_n = √(R² - 5s²)`, `R ≈ 1.2 A^(1/3)` fm, `s ≈ 0.9` fm.

**PySCF-DFT Approach** (more accurate):
1. Compute electron density `n(r)` from Kohn-Sham orbitals
2. Extract nuclear positions `R_I` from `cell.atom_coords()`
3. Fourier transform:
```python
F_nuc = np.sum([Z_I * np.exp(1j * q @ R_I) for I in range(N_atoms)])
```

### 2.3 Recoil Energy Spectrum

**Differential Rate**:
```
dR/dE_R = (ρ_χ σ_χN)/(2 m_χ m_N) × F²(E_R) × η(v_min, t)
```

**Astrophysical Factor**:
```
η(v_min, t) = ∫_{v>v_min} (f(v,t)/v) d³v
```
Where `v_min = √(m_N E_R / 2μ²)` and `f(v,t)` = Maxwell-Boltzmann.

**Parameters**:
- `ρ_χ = 0.3 GeV/cm³` (local DM density)
- `v_0 = 220 km/s` (Galactic rotation)
- `v_esc = 544 km/s` (Escape velocity)
- `v_earth = 232 km/s` (Earth's velocity, annual modulation)

### 2.4 DM-Electron Scattering (Low-Mass WIMPs)

For `m_χ < 1 GeV`, nuclear recoils are kinematically suppressed. Use electron recoils:

**Migdal Effect**: WIMP-nucleus scattering ionizes electrons
```
dR/dE_R dE_e = (dR/dE_R)_nuc × P_Migdal(E_R, E_e)
```

**Ionization Probability** (DFT):
```
P_ion(q) = |⟨ψ_f | e^(iq·r) | ψ_i⟩|²
```
Where `ψ_i` = valence band, `ψ_f` = conduction band (from PySCF).

**Crystal Form Factor**:
```
F_crys(q) = |∫_cell d³r n_val(r) e^(iq·r)|²
```
Computed by Fourier transforming valence density from DFT.

---

## 3. Axion Physics

### 3.1 Axion-Photon Coupling

**Lagrangian**:
```
L_aγγ = -(g_aγγ / 4) a F^μν F̃_μν
```
Where `F̃_μν = ε_μνρσ F^ρσ / 2` (dual field strength).

**Coupling Constant**:
```
g_aγγ = (α / 2π f_a) × (E/N - 1.92(4))
```
- `f_a` = Axion decay constant (~10⁹ GeV for DFSZ, ~10¹² GeV for KSVZ)
- `E/N` = Electromagnetic vs. color anomaly (model-dependent)

### 3.2 Primakoff Conversion in Crystals

**Process**: Axion + Virtual Photon → Real Photon
```
Γ_a→γ = (g_aγγ² m_a³)/(64π) × [1 - (m_γ/m_a)²]²
```

**Photon Effective Mass in Medium** (DFT):
```
m_γ² = ω_plasma² = (4π n_e α) / m_e
```
Where `n_e` = electron density from DFT.

**PySCF Calculation**:
```python
from pyscf.pbc import scf
mf = scf.RHF(cell).run()
n_e = mf.make_rdm1()  # Density matrix
rho_avg = np.mean(n_e)  # Average electron density
m_gamma_sq = 4 * np.pi * rho_avg * alpha / m_e
```

### 3.3 Crystal Electric Field

Axion-photon conversion enhanced by strong E-fields in crystal:

**Conversion Probability**:
```
P_a→γ = (g_aγγ B_T L)² × sin²(ΔkL/2) / (ΔkL/2)²
```
Where:
- `B_T` = Transverse magnetic field
- `L` = Crystal length
- `Δk = k_a - k_γ` = Momentum mismatch

**Electric Field from DFT**:
```
E_crystal(r) = -∇[V_H(r) + V_xc(r)]
```
Computed from DFT Hartree potential.

---

## 4. DarkPhoton Mixing

### 4.1 Kinetic Mixing Lagrangian

```
L = L_SM + L_DP - (ε/2) F^μν F'_μν
```
Where `F'_μν` = DarkPhoton field strength, `ε ~ 10⁻³ - 10⁻¹²`.

### 4.2 Mass Eigenstates

**Mass Matrix**:
```
M² = [ m_γ²,          ε m_γ m_A' ]
     [ ε m_γ m_A',    m_A'²      ]
```

**Diagonalization**:
```
tan(2θ) = (2 ε m_γ m_A') / (m_A'² - m_γ²)
```

**Physical Masses**:
```
m₁,₂² = (1/2)[m_γ² + m_A'² ∓ √((m_A'² - m_γ²)² + 4ε²m_γ²m_A'²)]
```

### 4.3 DFT Contribution: Photon Effective Mass

In-medium photon mass from plasma frequency:
```
m_γ(ω) = ω_p = √(4π n_e α / m_e)
```

**PySCF Implementation**:
```python
from pyscf.pbc import scf, dft

cell = build_silicon_cell()
mf = dft.RKS(cell, xc='PBE').run()
n_e = mf.get_rho()  # 3D electron density
omega_p = np.sqrt(4 * np.pi * n_e * alpha / m_e)
```

**Spatially-Varying ε**:
```
ε_eff(r) = ε_0 × [n_e(r) / n_e0]^(1/3)
```
Stronger mixing in high-density regions (e.g., near nuclei).

---

## 5. SterileNeutrino Scattering

### 5.1 Coherent Elastic Scattering

**Process**: ν_s + (Z,A) → ν_s + (Z,A)

**Differential Cross-Section**:
```
dσ/dE_R = (G_F² m_N)/(2π) × Q_w² × [1 - (m_N E_R)/(2E_ν²)] × F²(E_R)
```

Where:
- `G_F` = Fermi constant
- `Q_w = N - (1 - 4sin²θ_W)Z` = Weak charge
- `F²(E_R)` = Nuclear form factor (same as WIMP)

### 5.2 Sterile Mixing Angle

**Active-Sterile Mixing**:
```
|ν⟩ = cos(θ)|ν_active⟩ + sin(θ)|ν_sterile⟩
```

**Oscillation Probability**:
```
P(ν_e → ν_s) = sin²(2θ) sin²(1.27 Δm² L / E_ν)
```
- `Δm²` = Mass-squared difference [eV²]
- `L` = Baseline [m]
- `E_ν` = Neutrino energy [MeV]

### 5.3 DFT Contribution

Same nuclear form factor as WIMP case:
```
F²(q) = |∫ d³r ρ_nuc(r) e^(iq·r)|²
```

For coherent scattering, all nucleons add constructively → `F(0) = A`.

**PySCF Validation**:
Compare DFT form factor with Helm parameterization at low q.

---

## 6. Numerical Methods

### 6.1 Fourier Transform of Density

**Discrete Fourier Transform**:
```python
def form_factor(rho_r, q_vec, cell):
    """
    Compute F(q) = ∫ rho(r) exp(iq·r) dr

    Args:
        rho_r: Real-space density on grid [Nx, Ny, Nz]
        q_vec: Momentum transfer [qx, qy, qz] in inverse Bohr
        cell: PySCF Cell object

    Returns:
        F_q: Complex form factor
    """
    from pyscf.pbc.tools import fft

    # Real-space grid
    coords = cell.get_uniform_grids()

    # Phase factor
    phase = np.exp(1j * np.dot(coords, q_vec))

    # Integrate
    F_q = np.sum(rho_r * phase) * cell.vol / np.prod(rho_r.shape)

    return F_q
```

### 6.2 Momentum Transfer q-Scan

**Recoil Energy → Momentum**:
```
q = √(2 m_N E_R)
```

**Typical Range**:
- WIMP (m_χ ~ 100 GeV): `E_R ~ 1-100 keV` → `q ~ 0.01-0.5 GeV`
- Low-mass (m_χ ~ 1 GeV): `E_R ~ 1-100 eV` → `q ~ 0.001-0.05 GeV`

**Grid**:
```python
E_R = np.logspace(-3, 2, 100)  # 1 eV to 100 keV
q = np.sqrt(2 * m_nucleus * E_R) / hbar_c  # [GeV]
```

### 6.3 K-point Convergence

**Test Convergence**:
```python
for nk in [1, 2, 4, 8]:
    cell.kpts = cell.make_kpts([nk, nk, nk])
    mf = scf.RHF(cell).run()
    e_tot = mf.e_tot
    band_gap = compute_band_gap(mf)
    print(f"kpts={nk}³: E_tot={e_tot:.6f}, E_g={band_gap:.3f}")
```

**Convergence Criteria**:
- `|E_tot(k) - E_tot(k+1)| < 1e-4` Hartree
- `|E_g(k) - E_g(k+1)| < 0.01` eV

### 6.4 Exchange-Correlation Functionals

**LDA** (Local Density Approximation):
```
E_xc[n] = ∫ n(r) ε_xc(n(r)) dr
```
Fast but underestimates band gaps by ~50%.

**GGA-PBE** (Generalized Gradient):
```
E_xc[n] = ∫ n(r) ε_xc(n(r), ∇n(r)) dr
```
Better band gaps, recommended for DMCA.

**Hybrid HSE06**:
```
E_xc = (1/4)E_x^HF + (3/4)E_x^PBE + E_c^PBE
```
Accurate band gaps but 10x slower. Use for validation only.

### 6.5 Basis Set Convergence

**Plane Wave Energy Cutoff**:
```
E_cut = (1/2)|k + G_max|²
```

**Test Convergence**:
```python
for ecut in [30, 40, 50, 60, 80]:
    cell.basis = f'gth-dzvp-ecut{ecut}'
    mf = scf.RHF(cell).run()
    print(f"E_cut={ecut}: E_tot={mf.e_tot:.6f}")
```

**Recommended**: `E_cut = 50 Hartree` for Si/Ge (good balance).

---

## 7. References

### DFT & PySCF

1. **PySCF**: Sun et al., *J. Chem. Phys.* **153**, 024109 (2020)
2. **Kohn-Sham DFT**: Kohn & Sham, *Phys. Rev.* **140**, A1133 (1965)
3. **PBE Functional**: Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
4. **GTH Pseudopotentials**: Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996)

### WIMP Detection

5. **Lewin & Smith Review**: *Astropart. Phys.* **6**, 87 (1996)
6. **DM-Electron Scattering**: Essig et al., *JHEP* **05**, 046 (2016)
7. **Crystal Form Factors**: Trickle et al., *JHEP* **03**, 036 (2020)
8. **Migdal Effect**: Ibe et al., *JHEP* **03**, 194 (2018)

### Axion Physics

9. **ADMX Experiment**: Du et al., *Phys. Rev. Lett.* **120**, 151301 (2018)
10. **Axion-Photon Coupling**: Raffelt, *Lect. Notes Phys.* **741**, 51 (2008)
11. **Primakoff Process**: Sikivie, *Phys. Rev. Lett.* **51**, 1415 (1983)

### DarkPhoton

12. **Kinetic Mixing**: Holdom, *Phys. Lett. B* **166**, 196 (1986)
13. **Hidden Photon Searches**: Jaeckel & Ringwald, *Ann. Rev. Nucl. Part. Sci.* **60**, 405 (2010)

### SterileNeutrino

14. **Coherent Scattering**: Scholberg, *Phys. Rev. D* **73**, 033005 (2006)
15. **Sterile Neutrino Searches**: Gariazzo et al., *J. Phys. G* **43**, 033001 (2016)

---

**Next**: See `architecture-integration-design.md` for code-level implementation.
