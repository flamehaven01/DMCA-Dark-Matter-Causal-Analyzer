#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_hsta.py — SIDRCE HSTA (Holistic Systems Traceability Aggregation)

Computes final quality score Ω from three components:
- I (Integrity): Latest meta-pytest quality report
- P (Perfection): Final DFI-META evolution Omega
- D (Drift): NNSL methodology drift estimate

Formula: Ω = wI·I + wP·P + w1mD·(1-D)
Gate: PASS if Ω ≥ threshold (default: 0.90)
"""
import json
import glob
import os
import pathlib
import time
import sys
import yaml
import numpy as np


def load_latest_meta_quality(dirpath: str) -> float:
    """Load most recent meta-quality score from pytest reports."""
    pattern = os.path.join(dirpath, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)

    if not files:
        print(f"[WARN] No meta-quality reports found in {dirpath}, using I=0.0", file=sys.stderr)
        return 0.0

    latest_file = files[-1]
    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
        quality = float(data.get("quality", {}).get("Q", 0.0))
        print(f"[INFO] Loaded meta-quality: I = {quality:.4f} from {os.path.basename(latest_file)}")
        return quality
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[ERROR] Failed to parse {latest_file}: {e}", file=sys.stderr)
        return 0.0


def load_dfi_final_omega(path: str) -> float:
    """Load final DFI-META evolution Omega."""
    if not os.path.exists(path):
        print(f"[WARN] DFI-META results not found at {path}, using P=0.0", file=sys.stderr)
        return 0.0

    try:
        with open(path, "r") as f:
            data = json.load(f)
        omega = float(data.get("final_omega", 0.0))
        print(f"[INFO] Loaded DFI-META: P = {omega:.4f}")
        return omega
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[ERROR] Failed to parse {path}: {e}", file=sys.stderr)
        return 0.0


def nnsl_drift_estimate() -> float:
    """
    Estimate methodology drift from codebase structure.

    This is a synthetic estimator for demonstration purposes.
    Real implementation would analyze:
    - Code churn rates
    - API stability metrics
    - Dependency version drift

    Returns:
        float: Drift score in [0, 1] where 0 = stable, 1 = high drift
    """
    # Analyze source file structure as proxy for stability
    pattern = "src/**/*.py"
    paths = sorted(p for p in glob.glob(pattern, recursive=True))

    if not paths:
        print("[WARN] No source files found, using D=0.15 (default)", file=sys.stderr)
        return 0.15

    # Compute variance in file path lengths as drift proxy
    lengths = np.array([len(p) for p in paths], dtype=float)
    var = float(np.var(lengths)) if lengths.size else 0.0

    # Convert to drift metric (lower variance = more uniform = less drift)
    drift = 1.0 / (1.0 + var / 100.0)
    drift = 1.0 - drift  # Invert: high variance → high drift

    # Clamp to reasonable range
    drift = float(np.clip(drift, 0.0, 0.15))

    print(f"[INFO] NNSL drift estimate: D = {drift:.4f}")
    return drift


def main():
    """Main HSTA computation."""
    print("=" * 70)
    print("SIDRCE HSTA — Holistic Systems Traceability Aggregation")
    print("=" * 70)
    print()

    # Load configuration
    config_path = "configs/sidrce_hsta.yaml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Failed to parse YAML config: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract parameters
    wI = float(cfg["weights"]["wI"])
    wP = float(cfg["weights"]["wP"])
    w1mD = float(cfg["weights"]["w1mD"])
    PASS_THRESHOLD = float(cfg["gates"]["pass_threshold"])
    reports_dir = cfg["paths"]["meta_pytest_reports_dir"]
    dfi_path = cfg["paths"]["dfi_meta_result"]

    print(f"Weights: wI={wI}, wP={wP}, w1mD={w1mD}")
    print(f"Gate threshold: {PASS_THRESHOLD*100:.0f}%\n")

    # Compute components
    I = load_latest_meta_quality(reports_dir)
    P = load_dfi_final_omega(dfi_path)
    D = nnsl_drift_estimate()

    # Compute final Omega
    Omega = wI * I + wP * P + w1mD * (1.0 - D)
    passed = bool(Omega >= PASS_THRESHOLD)

    # Prepare result
    result = {
        "timestamp": time.time(),
        "components": {
            "I_integrity": round(I, 4),
            "P_perfection": round(P, 4),
            "D_drift": round(D, 4)
        },
        "weights": {
            "wI": wI,
            "wP": wP,
            "w1mD": w1mD
        },
        "Omega": round(Omega, 4),
        "Omega_pct": round(100.0 * Omega, 2),
        "pass_threshold_pct": round(100.0 * PASS_THRESHOLD, 2),
        "passed": passed
    }

    # Write report
    output_dir = pathlib.Path(".sidrce")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "hsta_report.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"I (Integrity):   {I:.4f} ({I*100:.1f}%)")
    print(f"P (Perfection):  {P:.4f} ({P*100:.1f}%)")
    print(f"D (Drift):       {D:.4f} ({D*100:.1f}%)")
    print()
    print(f"Ω = {wI}·I + {wP}·P + {w1mD}·(1-D)")
    print(f"Ω = {wI}·{I:.3f} + {wP}·{P:.3f} + {w1mD}·{1-D:.3f}")
    print(f"Ω = {Omega:.4f} ({Omega*100:.2f}%)")
    print()

    if passed:
        print(f"✓ PASS — Omega ({Omega*100:.2f}%) ≥ Threshold ({PASS_THRESHOLD*100:.0f}%)")
        print(f"\nReport saved to: {output_path}")
        sys.exit(0)
    else:
        print(f"✗ FAIL — Omega ({Omega*100:.2f}%) < Threshold ({PASS_THRESHOLD*100:.0f}%)")
        print(f"\nReport saved to: {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
