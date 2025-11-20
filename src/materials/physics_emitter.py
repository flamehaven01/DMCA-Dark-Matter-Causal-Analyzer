# -*- coding: utf-8 -*-
"""
src/materials/physics_emitter.py — Physics telemetry emitter.

Part of the DFI-Meta Meta-Cognitive Integration (Cycle 3).
Injects physics-specific metrics into the Sidrce telemetry stream.
"""
from typing import Dict, Any, Optional
import time
from ..sidrce_bridge import TelemetryEmitter

class PhysicsEmitter:
    """
    Wraps the core TelemetryEmitter to provide physics-aware logging.
    
    Attributes:
        emitter: Base TelemetryEmitter instance
    """
    
    def __init__(self, emitter: Optional[TelemetryEmitter] = None):
        self.emitter = emitter or TelemetryEmitter()

    def emit_physics(
        self,
        domain: str,
        hypothesis: str,
        step: int,
        physics_result: Dict[str, Any]
    ) -> None:
        """
        Emit a physics calculation result event.
        
        Args:
            domain: Problem domain (e.g., "dark_matter", "axion")
            hypothesis: Specific hypothesis being tested (e.g., "WIMP_Si_10GeV")
            step: Simulation step or iteration
            physics_result: Dictionary of physics metrics (e_tot, band_gap, etc.)
        """
        payload = {
            "event": "physics_calculation",
            "timestamp": time.time(),
            "domain": domain,
            "hypothesis": hypothesis,
            "step": step,
            "metrics": physics_result
        }
        
        # In a real implementation, we would validate the schema here
        # against the extended telemetry contract.
        
        self.emitter.emit(payload)

    def emit_material_selection(
        self,
        dm_mass: float,
        selected_material: str,
        reasoning: str
    ) -> None:
        """
        Emit a material selection event.
        
        Args:
            dm_mass: Dark Matter mass in MeV
            selected_material: Name of selected material (e.g., "Si")
            reasoning: Explanation for selection (e.g., "Gap match")
        """
        payload = {
            "event": "material_selection",
            "timestamp": time.time(),
            "dm_mass": dm_mass,
            "selected_material": selected_material,
            "reasoning": reasoning
        }
        
        self.emitter.emit(payload)
