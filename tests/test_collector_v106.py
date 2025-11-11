# -*- coding: utf-8 -*-
"""
Tests for SIDRCE Collector v1.0.6 - Data resilience and error handling
"""
import json
import tempfile
from pathlib import Path
import pytest
from sidrce.collector import (
    _iter_jsonl, summarize_telemetry, run_collect, calc_fti, lambda_effective
)


def test_iter_jsonl_resilience_corrupted_lines():
    """Test that corrupted JSONL lines are skipped gracefully"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Good line
        f.write(json.dumps({"domain": "test", "qps": 100}) + '\n')
        # Corrupted JSON
        f.write('{"invalid": json syntax}\n')
        # Empty line
        f.write('\n')
        # Another good line
        f.write(json.dumps({"domain": "test2", "qps": 200}) + '\n')
        # Incomplete JSON
        f.write('{"partial":\n')
        temp_path = f.name

    try:
        # Should yield only valid lines, skip corrupted ones
        rows = list(_iter_jsonl(Path(temp_path)))
        assert len(rows) == 2, f"Expected 2 valid rows, got {len(rows)}"
        assert rows[0]["qps"] == 100
        assert rows[1]["qps"] == 200
    finally:
        Path(temp_path).unlink()


def test_iter_jsonl_empty_file():
    """Test that empty file is handled gracefully"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name

    try:
        rows = list(_iter_jsonl(Path(temp_path)))
        assert len(rows) == 0
    finally:
        Path(temp_path).unlink()


def test_summarize_telemetry_valid_data():
    """Test FTI calculation with valid telemetry data"""
    rows = [
        {"qps": 480, "latency_p95_ms": 180, "infra_cost_usd": 120},
        {"qps": 510, "latency_p95_ms": 170, "infra_cost_usd": 102},
    ]

    result = summarize_telemetry(rows)

    assert result["count"] == 2
    assert "medians" in result
    assert "fti_avg" in result
    assert "fti_grade_worst" in result
    assert result["medians"]["qps"] == 495.0
    assert result["fti_avg"] > 0


def test_summarize_telemetry_empty_data():
    """Test that empty data returns valid defaults"""
    result = summarize_telemetry([])

    assert result["count"] == 0
    assert result["fti_avg"] == 0.0


def test_run_collect_missing_file():
    """Test that missing telemetry file returns error result instead of crashing"""
    result = run_collect(in_path="/nonexistent/path/ops_telemetry.jsonl")

    # Should return empty result with error, not crash
    assert result["count"] == 0
    assert "error" in result or "warning" in result
    assert result["fti_avg"] == 0.0


def test_run_collect_valid_data():
    """Test full collection pipeline with valid data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps({
            "domain": "direct_lab", "hypothesis": "WIMP", "step": 1,
            "intervention": "-", "infra_cost_usd": 120, "qps": 480, "latency_p95_ms": 180
        }) + '\n')
        f.write(json.dumps({
            "domain": "cosmology", "hypothesis": "DarkPhoton", "step": 1,
            "intervention": "-", "infra_cost_usd": 102, "qps": 510, "latency_p95_ms": 170
        }) + '\n')
        temp_in = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_out = f.name

    try:
        result = run_collect(in_path=temp_in, out_path=temp_out)

        assert result["count"] == 2
        assert result["fti_avg"] > 0
        assert result["fti_grade_worst"] in ["OK", "WARN", "FAIL"]
        assert "lambda_effective" in result

        # Verify output file was created
        assert Path(temp_out).exists()
        with open(temp_out) as f:
            saved = json.load(f)
            assert saved["count"] == 2
    finally:
        Path(temp_in).unlink()
        Path(temp_out).unlink()


def test_calc_fti_grades():
    """Test FTI grading logic"""
    # OK grade: FTI <= 1.0
    result = calc_fti(normalized_cost=0.9, normalized_perf=1.0)
    assert result.grade == "OK"
    assert result.fti <= 1.0

    # WARN grade: 1.0 < FTI <= 1.05
    result = calc_fti(normalized_cost=1.03, normalized_perf=1.0)
    assert result.grade == "WARN"
    assert 1.0 < result.fti <= 1.05

    # FAIL grade: FTI > 1.05
    result = calc_fti(normalized_cost=1.10, normalized_perf=1.0)
    assert result.grade == "FAIL"
    assert result.fti > 1.05


def test_lambda_effective_evolution():
    """Test lambda_effective calculation"""
    from sidrce.collector import LambdaParams

    params = LambdaParams()

    # Test normal scenario
    lam = lambda_effective(
        lambdacurrent=0.85,
        fticurrent=1.02,
        ftiref=1.0,
        deploy_rate_current=6.0,
        deploy_rate_ref=12.0,
        days_since_change=7.0,
        p=params
    )

    assert 0.0 < lam < 1.0, f"Lambda should be in (0,1), got {lam}"
    # Note: exponential decay can reduce lambda below sibg_min (by design)

    # Test that lambda decreases with time (exponential decay)
    lam_short = lambda_effective(
        lambdacurrent=0.85,
        fticurrent=1.02,
        ftiref=1.0,
        deploy_rate_current=6.0,
        deploy_rate_ref=12.0,
        days_since_change=1.0,  # Short time
        p=params
    )

    lam_long = lambda_effective(
        lambdacurrent=0.85,
        fticurrent=1.02,
        ftiref=1.0,
        deploy_rate_current=6.0,
        deploy_rate_ref=12.0,
        days_since_change=30.0,  # Long time
        p=params
    )

    # Lambda should decrease over time due to exponential decay
    assert lam_short > lam_long, f"Lambda should decrease over time: {lam_short} <= {lam_long}"
