import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath("src"))

from src.materials.physics_emitter import PhysicsEmitter

def test_emitter():
    print("Testing PhysicsEmitter...")
    
    # Mock the base emitter
    mock_base = MagicMock()
    emitter = PhysicsEmitter(emitter=mock_base)
    
    # Test emit_physics
    print("1. Testing emit_physics...")
    emitter.emit_physics(
        domain="dark_matter",
        hypothesis="WIMP_Si_10GeV",
        step=1,
        physics_result={"e_tot": -100.0, "gap": 1.12}
    )
    
    # Verify call
    mock_base.emit.assert_called()
    call_args = mock_base.emit.call_args[0][0]
    assert call_args["event"] == "physics_calculation"
    assert call_args["domain"] == "dark_matter"
    assert call_args["metrics"]["e_tot"] == -100.0
    print("   -> emit_physics PASSED")
    
    # Test emit_material_selection
    print("2. Testing emit_material_selection...")
    emitter.emit_material_selection(
        dm_mass=5.0,
        selected_material="Si",
        reasoning="Medium mass range"
    )
    
    # Verify call
    call_args = mock_base.emit.call_args[0][0]
    assert call_args["event"] == "material_selection"
    assert call_args["dm_mass"] == 5.0
    assert call_args["selected_material"] == "Si"
    print("   -> emit_material_selection PASSED")
    
    print("All PhysicsEmitter tests PASSED.")

if __name__ == "__main__":
    try:
        test_emitter()
    except Exception as e:
        print(f"PhysicsEmitter tests FAILED: {e}")
        sys.exit(1)
