import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath("src"))

# Mock PySCF before importing materials
sys.modules["pyscf"] = MagicMock()
sys.modules["pyscf.pbc"] = MagicMock()
sys.modules["pyscf.pbc.gto"] = MagicMock()
sys.modules["pyscf.pbc.dft"] = MagicMock()
sys.modules["pyscf.pbc.scf"] = MagicMock()

try:
    print("Attempting to import from src.materials...")
    from src.materials import (
        silicon, 
        germanium, 
        gallium_arsenide, 
        sodium_iodide, 
        cesium_iodide,
        select_material_by_dm_mass,
        MaterialModel
    )
    print("Imports successful.")

    # Mock the internal _pyscf_imports to return our mocks
    with patch("src.materials.base._pyscf_imports") as mock_imports:
        mock_gto = MagicMock()
        mock_dft = MagicMock()
        mock_scf = MagicMock()
        mock_imports.return_value = (mock_gto, mock_dft, mock_scf)
        
        # Mock Cell and KMF
        mock_cell = MagicMock()
        mock_gto.Cell.return_value = mock_cell
        mock_cell.make_kpts.return_value = [[0,0,0]]
        
        mock_kmf = MagicMock()
        mock_dft.KUKS.return_value = mock_kmf
        mock_kmf.density_fit.return_value = mock_kmf
        mock_kmf.converged = True

        print("Testing selection logic with mocks...")
        mat = select_material_by_dm_mass(0.5)
        print(f"Selected for 0.5 MeV: {mat}")
        
        mat = select_material_by_dm_mass(5.0)
        print(f"Selected for 5.0 MeV: {mat}")

    print("Refactor verification PASSED.")

except Exception as e:
    print(f"Refactor verification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
