# -*- coding: utf-8 -*-
"""
Tests for Causal Analysis v1.0.6 - Exception handling and configuration
"""
import os
import pytest
import pandas as pd
import numpy as np
from src.causal import (
    learn_structure_from_data, estimate_intervention_effect, _have_libs
)


def test_have_libs_detection():
    """Test library availability detection"""
    has_cl, has_dw = _have_libs()
    # Should return boolean tuple without crashing
    assert isinstance(has_cl, bool)
    assert isinstance(has_dw, bool)


def test_learn_structure_fallback_insufficient_data():
    """Test fallback to heuristic when data is insufficient"""
    # Small dataset should trigger fallback
    df = pd.DataFrame({
        'reliability_score': [0.9, 0.8],
        'sr9': [0.85, 0.75],
        'di2': [0.7, 0.6]
    })

    # With allow_fallback=True (default)
    result = learn_structure_from_data(df, allow_fallback=True, min_rows=1000)

    assert isinstance(result, str)
    assert 'digraph' in result
    # Should contain heuristic edges if columns match
    assert 'reliability_score' in result or 'sr9' in result


def test_learn_structure_fallback_disabled():
    """Test that RuntimeError is raised when fallback disabled"""
    df = pd.DataFrame({
        'x': [1, 2],
        'y': [3, 4]
    })

    has_cl, _ = _have_libs()
    if not has_cl:
        # Should raise when libraries unavailable and fallback disabled
        with pytest.raises(RuntimeError, match="SCM learning failed"):
            learn_structure_from_data(df, allow_fallback=False)


def test_estimate_intervention_effect_missing_columns():
    """Test that ValueError is raised for missing columns"""
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6]
    })
    graph_dot = "digraph G { }"

    # Treatment column doesn't exist
    with pytest.raises(ValueError, match="Treatment .* not in dataframe"):
        estimate_intervention_effect(df, graph_dot, treatment='nonexistent', outcome='y', intervention_value=10)

    # Outcome column doesn't exist
    with pytest.raises(ValueError, match="Outcome .* not in dataframe"):
        estimate_intervention_effect(df, graph_dot, treatment='x', outcome='nonexistent', intervention_value=10)


def test_estimate_intervention_effect_ols_fallback():
    """Test OLS fallback for causal estimation"""
    # Create simple linear relationship: y = 2x + noise
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = 2 * x + np.random.normal(0, 0.1, size=5)

    df = pd.DataFrame({'x': x, 'y': y})
    graph_dot = 'digraph G { "x" -> "y"; }'

    # Estimate effect of setting x to 10 (from mean ~3)
    effect = estimate_intervention_effect(
        df, graph_dot, treatment='x', outcome='y', intervention_value=10.0
    )

    # Effect should be approximately 2 * (10 - 3) = 14
    assert not np.isnan(effect), "Effect should not be NaN"
    assert 10 < effect < 20, f"Expected effect ~14, got {effect}"


def test_estimate_intervention_effect_insufficient_data():
    """Test handling of insufficient data (< 2 rows)"""
    df = pd.DataFrame({'x': [1], 'y': [2]})
    graph_dot = 'digraph G { "x" -> "y"; }'

    effect = estimate_intervention_effect(
        df, graph_dot, treatment='x', outcome='y', intervention_value=10.0
    )

    # Should return NaN for insufficient data
    assert np.isnan(effect)


def test_estimate_intervention_effect_configurable_method():
    """Test that DMCA_CAUSAL_METHOD environment variable is respected"""
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    graph_dot = 'digraph G { "x" -> "y"; }'

    # Set custom method
    original = os.environ.get('DMCA_CAUSAL_METHOD')
    try:
        os.environ['DMCA_CAUSAL_METHOD'] = 'backdoor.linear_regression'

        # Should not crash (method will be used if dowhy available, else OLS)
        effect = estimate_intervention_effect(
            df, graph_dot, treatment='x', outcome='y',
            intervention_value=1.0, method_name=None  # None triggers env var
        )

        assert isinstance(effect, float)
    finally:
        if original is not None:
            os.environ['DMCA_CAUSAL_METHOD'] = original
        elif 'DMCA_CAUSAL_METHOD' in os.environ:
            del os.environ['DMCA_CAUSAL_METHOD']


def test_learn_structure_with_sufficient_data():
    """Test structure learning with sufficient data"""
    # Generate sufficient data
    np.random.seed(42)
    n = 1100  # Above default MIN_ROWS threshold

    df = pd.DataFrame({
        'reliability_score': np.random.uniform(0.7, 1.0, n),
        'controversy_score': np.random.uniform(0.0, 0.5, n),
        'sr9': np.random.uniform(0.6, 0.9, n),
        'di2': np.random.uniform(0.5, 0.8, n)
    })

    # Should attempt PC algorithm if causallearn available, else fallback
    result = learn_structure_from_data(df, alpha=0.05, allow_fallback=True)

    assert isinstance(result, str)
    assert 'digraph' in result.lower()


def test_learn_structure_custom_min_rows():
    """Test custom min_rows parameter"""
    df = pd.DataFrame({
        'reliability_score': np.random.randn(50),
        'sr9': np.random.randn(50),
        'di2': np.random.randn(50)
    })

    # With min_rows=10, should attempt PC (if lib available)
    result = learn_structure_from_data(df, allow_fallback=True, min_rows=10)

    assert isinstance(result, str)
    assert 'digraph' in result

    # With min_rows=1000, should fallback
    result = learn_structure_from_data(df, allow_fallback=True, min_rows=1000)

    assert isinstance(result, str)
    assert 'digraph' in result
