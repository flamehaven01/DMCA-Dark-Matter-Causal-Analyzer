# -*- coding: utf-8 -*-
"""
src/materials/__init__.py — Package initialization.

Exposes the public API to maintain backward compatibility with the
original monolithic src/materials.py.
"""

from .base import (
    MaterialModel,
    build_cell,
    make_kpts,
    run_dft,
    recip_vectors,
    cell_volume,
    wrap_k_into_bz
)

from .silicon import silicon
from .germanium import germanium
from .gallium_arsenide import gallium_arsenide
from .sodium_iodide import sodium_iodide
from .cesium_iodide import cesium_iodide

from .selector import (
    select_material_by_dm_mass,
    list_available_materials
)

__all__ = [
    "MaterialModel",
    "build_cell",
    "make_kpts",
    "run_dft",
    "recip_vectors",
    "cell_volume",
    "wrap_k_into_bz",
    "silicon",
    "germanium",
    "gallium_arsenide",
    "sodium_iodide",
    "cesium_iodide",
    "select_material_by_dm_mass",
    "list_available_materials"
]
