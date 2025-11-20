# -*- coding: utf-8 -*-
"""
src/materials/sodium_iodide.py — Sodium Iodide material definition.
"""
import numpy as np
from .base import build_cell, make_kpts, run_dft, MaterialModel

def sodium_iodide(
    nk: tuple = (2, 2, 2),
    ecut: float = 10.0,
    xc: str = "PBE",
    use_hse: bool = False
) -> MaterialModel:
    """
    Sodium Iodide (NaI) Rock-salt Structure.
    a = 12.22 Bohr (6.47 Angstrom)
    """
    a = 12.22
    lattice = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]) * a
    
    atoms = [
        ('Na', (0.0, 0.0, 0.0)),
        ('I',  (0.5*a, 0.5*a, 0.5*a))
    ]
    
    cell = build_cell(lattice, atoms, mesh=(24, 24, 24))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc=xc, use_hse=use_hse)
    
    return MaterialModel(cell, kmf, kpts, functional="HSE06" if use_hse else xc)
