#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_astropy.py — Verify AstroPy integration and fallback logic.
"""
import sys
import os
from datetime import datetime
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.astro_analysis import get_earth_velocity_astropy, eta_shm_annual_modulation

class TestAstroPyIntegration(unittest.TestCase):
    
    def test_fallback_no_astropy(self):
        """Test fallback when astropy is missing."""
        with patch.dict(sys.modules, {'astropy': None}):
            # Force ImportError by mocking the import inside the function
            # Since we use lazy import, we need to patch builtins.__import__ or just rely on the fact 
            # that if we run this in an env without astropy it works.
            # But here we want to verify the logic even if astropy IS installed.
            
            # We'll mock the internal import by patching sys.modules to raise ImportError for astropy
            with patch.dict(sys.modules):
                sys.modules['astropy'] = None
                sys.modules['astropy.units'] = None
                sys.modules['astropy.coordinates'] = None
                sys.modules['astropy.time'] = None
                
                # We need to ensure the function re-imports (it's cached)
                get_earth_velocity_astropy.cache_clear()
                
                date = datetime(2025, 6, 2) # June 2 (peak velocity)
                v, method = get_earth_velocity_astropy(date)
                
                print(f"[Fallback Test] Velocity: {v:.2f} km/s, Method: {method}")
                self.assertIn("fallback", method)
                self.assertGreater(v, 240.0) # Should be near peak ~255

    def test_caching(self):
        """Test that caching works."""
        get_earth_velocity_astropy.cache_clear()
        date = datetime(2025, 1, 1)
        
        # First call
        v1, m1 = get_earth_velocity_astropy(date)
        
        # Second call
        v2, m2 = get_earth_velocity_astropy(date)
        
        self.assertEqual(v1, v2)
        self.assertEqual(m1, m2)
        
        # We can't easily check cache hits without inspecting the wrapper, 
        # but lru_cache is a standard library feature.
        info = get_earth_velocity_astropy.cache_info()
        print(f"[Cache Test] Hits: {info.hits}, Misses: {info.misses}")
        self.assertGreaterEqual(info.hits, 1)

    def test_astropy_logic_mock(self):
        """Test the 'astropy' path by mocking the astropy modules."""
        # We mock astropy to simulate it being present
        with patch.dict(sys.modules):
            mock_astropy = MagicMock()
            sys.modules['astropy'] = mock_astropy
            sys.modules['astropy.units'] = MagicMock()
            sys.modules['astropy.coordinates'] = MagicMock()
            sys.modules['astropy.time'] = MagicMock()
            
            get_earth_velocity_astropy.cache_clear()
            date = datetime(2025, 6, 2)
            
            # The function calls eta_shm_annual_modulation internally for now
            # so we expect it to return "astropy_date_derived"
            v, method = get_earth_velocity_astropy(date)
            
            print(f"[Mock AstroPy Test] Velocity: {v:.2f} km/s, Method: {method}")
            self.assertEqual(method, "astropy_date_derived")

if __name__ == "__main__":
    unittest.main()
