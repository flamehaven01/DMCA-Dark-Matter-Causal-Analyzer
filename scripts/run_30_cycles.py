#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_30_cycles.py — Orchestrate 30 DFI-META evolution + meta-pytest cycles

This script automates the NNSL × DFI-META × SIDRCE validation pipeline:
1. Run DFI-META evolution (methodology refinement)
2. Run meta-quality tests (pytest with quality metrics)
3. Repeat for 30 cycles
4. Compute final SIDRCE HSTA aggregation
"""
import subprocess
import sys
import time


def run(cmd, description=None):
    """Execute command with progress tracking."""
    if description:
        print(f"\n[RUN] {description}", flush=True)
    print(f"      $ {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] Command exited with code {result.returncode}", flush=True)
    return result


def main():
    """Main orchestration loop."""
    print("=" * 70)
    print("NNSL × DFI-META × SIDRCE — 30-Cycle Evolution Pipeline")
    print("=" * 70)
    print()

    start_time = time.time()

    # Run 30 evolution + test cycles
    for i in range(1, 31):
        cycle_start = time.time()
        print(f"\n{'='*70}")
        print(f"CYCLE {i}/30")
        print(f"{'='*70}")

        # Phase 1: DFI-META evolution
        run(
            [sys.executable, "scripts/dfi_meta_cli.py", "evolve", "--cycles", "1", "--threshold", "0.90"],
            description="DFI-META evolution (methodology refinement)"
        )

        # Phase 2: Meta-quality validation
        run(
            ["pytest", "-q", "tests/test_dm_physics_meta.py", "--disable-warnings"],
            description="Meta-quality validation (pytest)"
        )

        cycle_elapsed = time.time() - cycle_start
        print(f"\n[✓] Cycle {i} completed in {cycle_elapsed:.1f}s")

    # Phase 3: Final HSTA aggregation
    print(f"\n{'='*70}")
    print("FINAL AGGREGATION — SIDRCE HSTA")
    print(f"{'='*70}\n")

    result = run(
        [sys.executable, "scripts/compute_hsta.py"],
        description="Compute SIDRCE HSTA (Holistic Systems Traceability Aggregation)"
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}\n")

    # Exit with HSTA result code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
