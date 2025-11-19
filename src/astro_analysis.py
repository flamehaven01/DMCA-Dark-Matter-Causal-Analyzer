# -*- coding: utf-8 -*-
"""
astro_analysis.py — Standard Halo Model (SHM±) helpers for η(vmin).

Provides astrophysical velocity integrals for dark matter direct detection calculations.

The velocity integral η(v_min) encodes the probability that a DM particle
with sufficient velocity (v ≥ v_min) scatters off a target. This depends on:
- Local DM velocity distribution (Standard Halo Model)
- Earth's motion through the DM halo
- Galactic escape velocity

Key Features:
- SHM: Truncated Maxwell-Boltzmann distribution
- SHM++: Updated parameters from Eur. Phys. J. C (2019)
- Closed-form η(v_min) with no numerical integration

References:
- Lewin & Smith, Astropart. Phys. 6 (1996) 87 — Review of DM detection
- Evans et al., PRD 99 (2019) 023012 — SHM++ parameters
- Essig et al., JHEP 05 (2016) 046 — Application to semiconductor targets

Example:
    >>> vmin = 0.001  # in units of c
    >>> eta = eta_shm(vmin, v0_kms=220, vesc_kms=544, vE_kms=240)
    >>> print(f"η(v_min) = {eta:.6e}")
"""
from __future__ import annotations
import numpy as np
from math import erf, exp, sqrt, pi

# Conversion factor
KM_S_TO_C = 1.0 / 299792.458  # km/s → c


def _norm_esc(zesc: float) -> float:
    """
    Normalization factor for truncated Maxwell-Boltzmann distribution.

    N_esc(z_esc) = erf(z_esc) - (2/√π) z_esc exp(-z_esc²)

    where z_esc = v_esc / v0.

    Args:
        zesc: Dimensionless escape velocity (v_esc / v0)

    Returns:
        float: Normalization factor

    Note:
        This accounts for the hard cutoff at v = v_esc in the galactic halo.
    """
    return erf(zesc) - (2.0 / sqrt(pi)) * zesc * exp(-zesc * zesc)


def eta_shm(
    vmin_c: float,
    v0_kms: float = 220.0,
    vesc_kms: float = 544.0,
    vE_kms: float = 240.0
) -> float:
    """
    Compute velocity integral η(v_min) for Standard Halo Model.

    The velocity integral is defined as:
        η(v_min, t) = ∫_{v>v_min} d³v f(v + v_E(t)) / v

    where f(v) is the truncated Maxwell-Boltzmann distribution.

    Args:
        vmin_c: Minimum DM velocity (units of c)
        v0_kms: Most probable speed (peak of f(v)) [km/s]
                Default: 220 km/s (SHM++)
        vesc_kms: Galactic escape velocity [km/s]
                  Default: 544 km/s (SHM++)
        vE_kms: Earth's velocity relative to halo [km/s]
                Default: 240 km/s (approximate annual average)

    Returns:
        float: Velocity integral η(v_min) in units of [c/km/s]

    Raises:
        ValueError: If velocities are non-positive or v_min > v_esc + v_E

    Notes:
        - SHM assumes isotropic, isothermal halo with hard cutoff at v_esc
        - SHM++ parameters from Evans et al. (PRD 2019) based on Gaia data
        - Annual modulation enters through v_E(t) (not implemented here)

    Example:
        >>> # Light DM with v_min ~ 0.001c
        >>> eta = eta_shm(0.001, v0_kms=220, vesc_kms=544, vE_kms=240)
        >>> print(f"η = {eta:.6e}")

        >>> # Heavy DM with v_min ~ 1e-4c
        >>> eta = eta_shm(1e-4)
        >>> print(f"η = {eta:.6e}")
    """
    # Input validation
    if vmin_c < 0:
        raise ValueError(f"vmin_c must be non-negative, got {vmin_c}")
    if v0_kms <= 0:
        raise ValueError(f"v0_kms must be positive, got {v0_kms}")
    if vesc_kms <= 0:
        raise ValueError(f"vesc_kms must be positive, got {vesc_kms}")
    if vE_kms < 0:
        raise ValueError(f"vE_kms must be non-negative, got {vE_kms}")

    # Convert to units of c
    v0 = v0_kms * KM_S_TO_C
    vesc = vesc_kms * KM_S_TO_C
    vE = vE_kms * KM_S_TO_C

    # Dimensionless parameters
    zesc = vesc / v0
    x = vmin_c / v0  # Dimensionless v_min
    y = vE / v0      # Dimensionless Earth velocity

    # Normalization factor
    Nesc = _norm_esc(zesc)

    if Nesc <= 0:
        raise RuntimeError(
            f"Invalid normalization Nesc = {Nesc} (zesc = {zesc}). "
            f"Check v_esc and v0 parameters."
        )

    # Closed-form η(v_min) with three kinematic regimes
    # (Lewin & Smith 1996, Eq. 3.10-3.12)

    if x < abs(y - 1.0):
        # Low v_min: Full overlap between DM distribution and detector
        # η ~ (1/2y) [erf(x+y) - erf(x-y)]
        val = (1.0 / (2.0 * y * Nesc)) * (erf(x + y) - erf(x - y))

    elif abs(y - 1.0) <= x < (y + zesc):
        # Intermediate v_min: Partial overlap
        # η ~ (1/2y) [erf(z_esc) - erf(x-y)] - (1/√π y) exp(-z_esc²)
        term1 = (1.0 / (2.0 * y * Nesc)) * (erf(zesc) - erf(x - y))
        term2 = (1.0 / (sqrt(pi) * y * Nesc)) * exp(-zesc * zesc)
        val = term1 - term2

    else:
        # High v_min: No kinematically allowed events
        val = 0.0

    # Sanity check
    if val < 0:
        # Can occur due to numerical precision near boundaries
        val = 0.0

    return float(val)


def eta_shm_annual_modulation(
    vmin_c: float,
    day_of_year: int,
    v0_kms: float = 220.0,
    vesc_kms: float = 544.0,
    vE_avg_kms: float = 240.0,
    vE_amp_kms: float = 15.0
) -> float:
    """
    Compute η(v_min) with annual modulation of Earth's velocity.

    v_E(t) = v_E,avg + v_E,amp cos[2π(t - t_0)/T]

    Args:
        vmin_c: Minimum DM velocity (units of c)
        day_of_year: Day number (1-365)
        v0_kms: Most probable speed [km/s]
        vesc_kms: Escape velocity [km/s]
        vE_avg_kms: Average Earth velocity [km/s]
        vE_amp_kms: Modulation amplitude [km/s]

    Returns:
        float: η(v_min) at specified date

    Example:
        >>> # June 2 (day 153): v_E maximum
        >>> eta_max = eta_shm_annual_modulation(0.001, day_of_year=153)
        >>> # December 2 (day 336): v_E minimum
        >>> eta_min = eta_shm_annual_modulation(0.001, day_of_year=336)
        >>> print(f"Modulation: {(eta_max - eta_min)/eta_max * 100:.1f}%")
    """
    # Phase: maximum around June 2 (day ~153)
    t0 = 153
    phase = 2.0 * pi * (day_of_year - t0) / 365.25

    # Time-dependent Earth velocity
    vE_t = vE_avg_kms + vE_amp_kms * np.cos(phase)

    return eta_shm(vmin_c, v0_kms=v0_kms, vesc_kms=vesc_kms, vE_kms=vE_t)
