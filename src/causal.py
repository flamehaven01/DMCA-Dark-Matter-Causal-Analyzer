# -*- coding: utf-8 -*-
"""Causal Analysis: PC/DoWhy with Hybrid SCM Fallback (no hard fail)"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np

# Configurable threshold: default 1000 rows
_MIN_ROWS = int(os.environ.get("DMCA_MIN_ROWS_FOR_DISCOVERY", "1000"))

_DEF_HEURISTIC_EDGES = (
    ('reliability_score', 'sr9'),
    ('controversy_score', 'sr9'),
    ('sr9', 'di2'),
)

def _have_libs() -> tuple[bool, bool]:
    try:
        import causallearn  # noqa
        has_cl = True
    except Exception:
        has_cl = False
    try:
        import dowhy  # noqa
        has_dw = True
    except Exception:
        has_dw = False
    return has_cl, has_dw

def learn_structure_from_data(df: pd.DataFrame, alpha: float = 0.05,
                              allow_fallback: bool = True,
                              min_rows: int = None) -> str:
    min_rows = _MIN_ROWS if min_rows is None else min_rows
    has_cl, _ = _have_libs()
    if has_cl and len(df) >= min_rows:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cg = pc(df.values, alpha, fisherz)
        try:
            return cg.G.graph_to_dot(node_labels=df.columns)
        except Exception:
            edges = []
            for (i, j) in cg.G.get_graph_edges():
                a = df.columns[i]
                b = df.columns[j]
                edges.append(f'"{a}" -> "{b}";')
            return "digraph G { " + " ".join(edges) + " }"
    if allow_fallback:
        cols = set(df.columns)
        edges = [f'"{a}" -> "{b}";' for (a, b) in _DEF_HEURISTIC_EDGES if {a, b}.issubset(cols)]
        return "digraph G { " + " ".join(edges) + " }"
    raise RuntimeError("SCM learning failed and fallback disabled")

def estimate_intervention_effect(df: pd.DataFrame, graph_dot: str,
                                 treatment: str, outcome: str,
                                 intervention_value: float) -> float:
    has_cl, has_dw = _have_libs()
    if has_dw and len(df) >= max(10, int(0.1 * _MIN_ROWS)):
        from dowhy import CausalModel
        model = CausalModel(
            data=df, treatment=treatment, outcome=outcome,
            graph=graph_dot.replace('\n', ' ')
        )
        estimand = model.identify_effect()
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=float(df[treatment].mean()),
            treatment_value=float(intervention_value),
        )
        return float(estimate.value)
    # OLS fallback (slope * Δtreatment)
    x = df[treatment].to_numpy()
    y = df[outcome].to_numpy()
    if len(x) < 2:
        return float("nan")
    X = np.vstack([x, np.ones_like(x)]).T
    beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    delta = float(intervention_value) - float(np.mean(x))
    return float(beta * delta)
