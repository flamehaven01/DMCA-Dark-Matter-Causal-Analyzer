# -*- coding: utf-8 -*-
"""
src/materials/selector.py — Material selection logic.
"""
from typing import List, Optional, Tuple, Union, Dict, Any
from .base import MaterialModel
from .silicon import silicon
from .germanium import germanium
from .gallium_arsenide import gallium_arsenide
from .sodium_iodide import sodium_iodide
from .cesium_iodide import cesium_iodide

MATERIALS_DB = {
    "Si": {"gap_eV": 1.12, "excitonic": False},
    "Ge": {"gap_eV": 0.67, "excitonic": False},
    "GaAs": {"gap_eV": 1.42, "excitonic": False},
    "NaI": {"gap_eV": 5.9, "excitonic": True},
    "CsI": {"gap_eV": 6.1, "excitonic": True}
}

def list_available_materials() -> Dict[str, Any]:
    """Return dictionary of supported materials with metadata."""
    return MATERIALS_DB

def select_material_by_dm_mass(
    mchi_GeV: float,
    prioritize_excitons: bool = False
) -> Tuple[Union[MaterialModel, str], str]:
    """
    Select optimal material based on Dark Matter mass.
    
    Strategy (Legacy):
    - Low mass (< 1 GeV): NaI (if excitons) or Si
    - Medium mass (1-10 GeV): Si
    - High mass (> 10 GeV): Ge (Coherent scattering enhancement)
    
    Args:
        mchi_GeV: DM mass in GeV
        prioritize_excitons: Prefer materials with excitonic effects (NaI, CsI)
        
    Returns:
        Tuple[MaterialModel, str]: (Initialized material model, Rationale string)
    """
    rationale = ""
    
    if prioritize_excitons:
        if mchi_GeV < 10.0:
            # NaI is good for excitons even at lower masses if prioritized
            rationale = f"Selected NaI for excitonic sensitivity (mass {mchi_GeV} GeV)"
            try:
                return sodium_iodide(), rationale
            except ImportError:
                return "NaI", rationale
        else:
            rationale = f"Selected CsI for high-mass excitonic sensitivity (mass {mchi_GeV} GeV)"
            try:
                return cesium_iodide(), rationale
            except ImportError:
                return "CsI", rationale

    if mchi_GeV < 1.0:
        # Low mass target
        rationale = f"Selected Si for low mass {mchi_GeV} GeV (Standard)"
        try:
            return silicon(), rationale
        except ImportError:
            return "Si", rationale
    elif mchi_GeV < 10.0:
        # Medium mass target: Silicon
        rationale = f"Selected Si for standard sensitivity (1.12 eV gap) at mass {mchi_GeV} GeV"
        try:
            return silicon(), rationale
        except ImportError:
            return "Si", rationale
    else:
        # High mass target (> 10 GeV): Germanium (Heavy nucleus, A^2 enhancement)
        rationale = f"Selected Ge for high mass {mchi_GeV} GeV (Coherent scattering)"
        try:
            return germanium(), rationale
        except ImportError:
            return "Ge", rationale
