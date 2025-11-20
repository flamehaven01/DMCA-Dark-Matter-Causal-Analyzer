# -*- coding: utf-8 -*-
"""
src/materials/gallium_arsenide.py — Gallium Arsenide material definition.
"""
import numpy as np
from .base import build_cell, make_kpts, run_dft, MaterialModel

def gallium_arsenide(
    nk: tuple = (4, 4, 4),
    ecut: float = 10.0,
    xc: str = "PBE",
    use_hse: bool = False
) -> MaterialModel:
    """
    Gallium Arsenide (GaAs) Zincblende Structure.
    a = 10.68 Bohr (5.65 Angstrom)
    """
    a = 10.68
    lattice = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]) * a
    
    atoms = [
        ('Ga', (0.0, 0.0, 0.0)),
        ('As', (a/4, a/4, a/4))
    ]
    
    cell = build_cell(lattice, atoms, mesh=(24, 24, 24))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc=xc, use_hse=use_hse)
    
    return MaterialModel(cell, kmf, kpts, functional="HSE06" if use_hse else xc)
