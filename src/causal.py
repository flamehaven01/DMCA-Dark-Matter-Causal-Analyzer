# -*- coding: utf-8 -*-
"""Causal Analysis: PC/DoWhy with Hybrid SCM Fallback (no hard fail)"""
from __future__ import annotations
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Configurable threshold: default 1000 rows
_MIN_ROWS = int(os.environ.get("DMCA_MIN_ROWS_FOR_DISCOVERY", "1000"))

# Configurable inference method for DoWhy
_CAUSAL_METHOD = os.environ.get("DMCA_CAUSAL_METHOD", "backdoor.linear_regression")

_DEF_HEURISTIC_EDGES = (
    ('reliability_score', 'sr9'),
    ('controversy_score', 'sr9'),
    ('sr9', 'di2'),
)

def _have_libs() -> tuple[bool, bool]:
    try:
        import causallearn  # noqa
        has_cl = True
    except ImportError:
        has_cl = False
    try:
        import dowhy  # noqa
        has_dw = True
    except ImportError:
        has_dw = False
    return has_cl, has_dw

def learn_structure_from_data(df: pd.DataFrame, alpha: float = 0.05,
                              allow_fallback: bool = True,
                              min_rows: int = None) -> str:
    """
    Learn causal graph structure from data using PC algorithm or fallback.

    Args:
        df: Input dataframe
        alpha: Significance level for conditional independence tests
        allow_fallback: Use heuristic edges if library unavailable
        min_rows: Minimum rows required for PC algorithm

    Returns:
        DOT format graph string

    Raises:
        RuntimeError: If learning fails and fallback disabled
    """
    min_rows = _MIN_ROWS if min_rows is None else min_rows
    has_cl, _ = _have_libs()

    if has_cl and len(df) >= min_rows:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        try:
            cg = pc(df.values, alpha, fisherz)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"PC algorithm failed: {e}. Falling back to heuristic.")
            if allow_fallback:
                cols = set(df.columns)
                edges = [f'"{a}" -> "{b}";' for (a, b) in _DEF_HEURISTIC_EDGES if {a, b}.issubset(cols)]
                return "digraph G { " + " ".join(edges) + " }"
            raise

        # Try to convert to DOT format
        try:
            return cg.G.graph_to_dot(node_labels=df.columns)
        except AttributeError as e:
            # graph_to_dot not available, construct manually
            logger.debug(f"graph_to_dot failed: {e}. Constructing edges manually.")
            edges = []
            try:
                for (i, j) in cg.G.get_graph_edges():
                    a = df.columns[i]
                    b = df.columns[j]
                    edges.append(f'"{a}" -> "{b}";')
                return "digraph G { " + " ".join(edges) + " }"
            except Exception as e:
                logger.warning(f"Failed to extract edges: {e}. Falling back to heuristic.")
                if allow_fallback:
                    cols = set(df.columns)
                    edges = [f'"{a}" -> "{b}";' for (a, b) in _DEF_HEURISTIC_EDGES if {a, b}.issubset(cols)]
                    return "digraph G { " + " ".join(edges) + " }"
                raise

    if allow_fallback:
        logger.info("Using heuristic edges (causallearn unavailable or insufficient data)")
        cols = set(df.columns)
        edges = [f'"{a}" -> "{b}";' for (a, b) in _DEF_HEURISTIC_EDGES if {a, b}.issubset(cols)]
        return "digraph G { " + " ".join(edges) + " }"

    raise RuntimeError("SCM learning failed and fallback disabled")

def estimate_intervention_effect(df: pd.DataFrame, graph_dot: str,
                                 treatment: str, outcome: str,
                                 intervention_value: float,
                                 method_name: str = None) -> float:
    """
    Estimate causal effect of intervention using DoWhy or OLS fallback.

    Args:
        df: Input dataframe
        graph_dot: DOT format causal graph
        treatment: Treatment variable name
        outcome: Outcome variable name
        intervention_value: Value to set treatment to
        method_name: Estimation method (default from env or 'backdoor.linear_regression')

    Returns:
        Estimated causal effect

    Raises:
        ValueError: If columns not found in dataframe
    """
    if treatment not in df.columns:
        raise ValueError(f"Treatment '{treatment}' not in dataframe columns")
    if outcome not in df.columns:
        raise ValueError(f"Outcome '{outcome}' not in dataframe columns")

    if method_name is None:
        method_name = _CAUSAL_METHOD

    has_cl, has_dw = _have_libs()

    if has_dw and len(df) >= max(10, int(0.1 * _MIN_ROWS)):
        from dowhy import CausalModel

        try:
            model = CausalModel(
                data=df, treatment=treatment, outcome=outcome,
                graph=graph_dot.replace('\n', ' ')
            )
            estimand = model.identify_effect()
            estimate = model.estimate_effect(
                estimand,
                method_name=method_name,
                control_value=float(df[treatment].mean()),
                treatment_value=float(intervention_value),
            )
            logger.info(f"DoWhy estimate: {estimate.value:.4f} using {method_name}")
            return float(estimate.value)
        except (ValueError, RuntimeError, KeyError) as e:
            logger.warning(f"DoWhy estimation failed: {e}. Falling back to OLS.")
            # Fall through to OLS fallback

    # OLS fallback (slope * Δtreatment)
    logger.debug("Using OLS fallback for causal estimation")
    x = df[treatment].to_numpy()
    y = df[outcome].to_numpy()

    if len(x) < 2:
        logger.warning("Insufficient data for causal estimation (< 2 rows)")
        return float("nan")

    X = np.vstack([x, np.ones_like(x)]).T
    try:
        beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        delta = float(intervention_value) - float(np.mean(x))
        effect = float(beta * delta)
        logger.debug(f"OLS estimate: {effect:.4f} (beta={beta:.4f}, delta={delta:.4f})")
        return effect
    except np.linalg.LinAlgError as e:
        logger.error(f"OLS estimation failed: {e}")
        return float("nan")
