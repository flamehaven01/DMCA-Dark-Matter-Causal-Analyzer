# -*- coding: utf-8 -*-
"""
SIDRCE Collector — FTI v2 & λ_effective from ops_telemetry.jsonl
DMCA는 FTI/λ를 계산하지 않습니다. 이 모듈이 외부에서 계산합니다.
"""
from __future__ import annotations
import json, math, statistics as stats, base64, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except Exception:
    _HAS_CRYPTO = False

@dataclass
class FTIResult:
    fti: float
    grade: str  # OK / WARN / FAIL

def _norm_perf(qps: float, qps_med: float, lat_ms: float, lat_med: float) -> float:
    # Geometric blend of (QPS ↑) and (1/Latency ↑)
    return ((qps / max(1e-9, qps_med)) *
            ((1.0/max(1e-9, lat_ms)) / (1.0/max(1e-9, lat_med)))) ** 0.5

def _norm_cost(cost: float, cost_med: float) -> float:
    return cost / max(1e-9, cost_med)

def calc_fti(normalized_cost: float, normalized_perf: float) -> FTIResult:
    fti = float(normalized_cost) / max(1e-9, float(normalized_perf))
    grade = "OK" if fti <= 1.0 else ("WARN" if fti <= 1.05 else "FAIL")
    return FTIResult(fti=fti, grade=grade)

@dataclass
class LambdaParams:
    beta: float = 0.33
    alpha: float = 0.5
    lambda0: float = 0.10
    sibg_min: float = 0.70
    sibg_max: float = 0.98
    asdpi_min: float = 0.03
    asdpi_max: float = 0.25

def _clamp(x, lo, hi): return max(lo, min(hi, x))

def lambda_effective(lambdacurrent: float, fticurrent: float, ftiref: float,
                     deploy_rate_current: float, deploy_rate_ref: float,
                     days_since_change: float, p: LambdaParams = LambdaParams()) -> float:
    # SIBG: pressure to evolve from FinOps tradeoff
    ratio = fticurrent / max(1e-9, ftiref)
    lambdasibg = _clamp(lambdacurrent + p.beta * (ratio - 1.0), p.sibg_min, p.sibg_max)
    # ASDPI: governance damping (slower/safer deploys -> stronger damping)
    ratio_d = deploy_rate_ref / max(0.01, deploy_rate_current)
    lambdaasdpi = _clamp(p.lambda0 + p.alpha * (ratio_d - 1.0), p.asdpi_min, p.asdpi_max)
    return lambdasibg * math.exp(-lambdaasdpi * days_since_change)

def _maybe_decrypt(obj: Dict[str, Any], key: bytes = None) -> Dict[str, Any]:
    if not obj or not obj.get("enc"):
        return obj
    if not (_HAS_CRYPTO and key):
        raise RuntimeError("encrypted telemetry present but no decryption key configured")
    aes = AESGCM(key)
    nonce = base64.b64decode(obj["nonce"])
    aad = base64.b64decode(obj.get("aad", ""))
    ct = base64.b64decode(obj["ct"])
    pt = aes.decrypt(nonce, ct, aad)
    return json.loads(pt.decode("utf-8"))

def _iter_jsonl(path: Path, key: bytes = None) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL with error resilience (skip corrupted lines)"""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield _maybe_decrypt(obj, key)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted line {i} in {path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process line {i} in {path}: {e}")
                continue

def summarize_telemetry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    qps_list = [float(r["qps"]) for r in rows if "qps" in r]
    lat_list = [float(r["latency_p95_ms"]) for r in rows if "latency_p95_ms" in r]
    cost_list = [float(r["infra_cost_usd"]) for r in rows if "infra_cost_usd" in r]

    qps_med = stats.median(qps_list) if qps_list else 1.0
    lat_med = stats.median(lat_list) if lat_list else 1.0
    cost_med = stats.median(cost_list) if cost_list else 1.0

    fti_values: List[float] = []
    grades: List[str] = []
    for r in rows:
        npf = _norm_perf(float(r["qps"]), qps_med, float(r["latency_p95_ms"]), lat_med)
        nct = _norm_cost(float(r["infra_cost_usd"]), cost_med)
        res = calc_fti(nct, npf)
        fti_values.append(res.fti)
        grades.append(res.grade)

    fti_avg = sum(fti_values) / max(1, len(fti_values))
    worst = "FAIL" if "FAIL" in grades else ("WARN" if "WARN" in grades else "OK")

    return {
        "count": len(rows),
        "medians": {"qps": qps_med, "latency_p95_ms": lat_med, "infra_cost_usd": cost_med},
        "fti_avg": fti_avg,
        "fti_grade_worst": worst
    }

def run_collect(in_path: str = "ops_telemetry.jsonl",
                out_path: str = "sidrce_report.json") -> Dict[str, Any]:
    import os
    key_env = os.environ.get("SIDRCE_AES_KEY", "")
    key = base64.b64decode(key_env) if key_env else None

    p = Path(in_path)
    if not p.exists():
        logger.error(f"Telemetry file not found: {p}")
        # Return empty result instead of crashing
        return {
            "count": 0,
            "medians": {"qps": 1.0, "latency_p95_ms": 1.0, "infra_cost_usd": 1.0},
            "fti_avg": 0.0,
            "fti_grade_worst": "OK",
            "lambda_effective": 0.0,
            "error": f"File not found: {p}"
        }

    rows = list(_iter_jsonl(p, key))

    if not rows:
        logger.warning(f"No valid telemetry rows in {p}")
        return {
            "count": 0,
            "medians": {"qps": 1.0, "latency_p95_ms": 1.0, "infra_cost_usd": 1.0},
            "fti_avg": 0.0,
            "fti_grade_worst": "OK",
            "lambda_effective": 0.0,
            "warning": "No valid telemetry data"
        }

    summary = summarize_telemetry(rows)

    # Example λ calc (static params, illustrative)
    lam = lambda_effective(
        lambdacurrent=0.85,
        fticurrent=summary["fti_avg"],
        ftiref=1.0,
        deploy_rate_current=6.0,  # weekly deploys
        deploy_rate_ref=12.0,     # ref pace
        days_since_change=7.0
    )
    payload = {**summary, "lambda_effective": lam}

    try:
        Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write report to {out_path}: {e}")

    logger.info(f"Collected telemetry: {summary['count']} rows, FTI={summary['fti_avg']:.3f}, λ={lam:.3f}")
    return payload

if __name__ == "__main__":
    run_collect()
