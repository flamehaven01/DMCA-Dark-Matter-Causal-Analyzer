# -*- coding: utf-8 -*-
"""
src/materials/germanium.py — Germanium material definition.
"""
import numpy as np
from .base import build_cell, make_kpts, run_dft, MaterialModel

def germanium(
    nk: tuple = (4, 4, 4),
    ecut: float = 10.0,
    xc: str = "PBE",
    use_hse: bool = False
) -> MaterialModel:
    """
    Germanium (Ge) Diamond Structure.
    a = 10.69 Bohr (5.66 Angstrom)
    """
    a = 10.69
    lattice = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]) * a
    
    atoms = [
        ('Ge', (0.0, 0.0, 0.0)),
        ('Ge', (a/4, a/4, a/4))
    ]
    
    cell = build_cell(lattice, atoms, mesh=(24, 24, 24))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc=xc, use_hse=use_hse)
    
    return MaterialModel(cell, kmf, kpts, functional="HSE06" if use_hse else xc)
