# -*- coding: utf-8 -*-
"""
src/materials/cesium_iodide.py — Cesium Iodide material definition.
"""
import numpy as np
from .base import build_cell, make_kpts, run_dft, MaterialModel

def cesium_iodide(
    nk: tuple = (2, 2, 2),
    ecut: float = 10.0,
    xc: str = "PBE",
    use_hse: bool = False
) -> MaterialModel:
    """
    Cesium Iodide (CsI) Simple Cubic (CsCl structure).
    a = 8.63 Bohr (4.57 Angstrom)
    """
    a = 8.63
    lattice = np.diag([a, a, a])
    
    atoms = [
        ('Cs', (0.0, 0.0, 0.0)),
        ('I',  (0.5*a, 0.5*a, 0.5*a))
    ]
    
    cell = build_cell(lattice, atoms, mesh=(24, 24, 24))
    kpts = make_kpts(cell, nk)
    kmf = run_dft(cell, kpts, xc=xc, use_hse=use_hse)
    
    return MaterialModel(cell, kmf, kpts, functional="HSE06" if use_hse else xc)
