# -*- coding: utf-8 -*-
"""SIDRCE Server — FastAPI /report & /metrics with background caching"""
from __future__ import annotations
import json, os, logging, time
from threading import Thread, Lock
from typing import Dict, Any
from fastapi import FastAPI, Response
from .collector import run_collect

logger = logging.getLogger(__name__)

app = FastAPI(title="SIDRCE Server", version="1.0.6")

# Global cache with thread-safe lock
_LATEST_PAYLOAD: Dict[str, Any] = {}
_CACHE_LOCK = Lock()
_CACHE_UPDATE_INTERVAL = float(os.environ.get("SIDRCE_CACHE_INTERVAL", "30"))  # seconds

def background_collector():
    """Background thread that periodically updates metrics cache"""
    logger.info(f"[SIDRCE] Background collector started (interval: {_CACHE_UPDATE_INTERVAL}s)")

    while True:
        try:
            # Run collection
            payload = run_collect(
                in_path=os.environ.get("SIDRCE_STREAM", "ops_telemetry.jsonl"),
                out_path=os.environ.get("SIDRCE_REPORT", "sidrce_report.json")
            )

            # Update cache atomically
            with _CACHE_LOCK:
                global _LATEST_PAYLOAD
                _LATEST_PAYLOAD = payload
                _LATEST_PAYLOAD['cache_timestamp'] = time.time()

            logger.debug(f"[SIDRCE] Cache updated: FTI={payload.get('fti_avg', 0):.3f}, λ={payload.get('lambda_effective', 0):.3f}")

        except Exception as e:
            logger.error(f"[SIDRCE] Background collector error: {e}", exc_info=True)

        # Sleep until next update
        time.sleep(_CACHE_UPDATE_INTERVAL)

@app.on_event("startup")
def startup_event():
    """Initialize background collector on server start"""
    # Initial cache population
    try:
        payload = run_collect(
            in_path=os.environ.get("SIDRCE_STREAM", "ops_telemetry.jsonl"),
            out_path=os.environ.get("SIDRCE_REPORT", "sidrce_report.json")
        )
        with _CACHE_LOCK:
            global _LATEST_PAYLOAD
            _LATEST_PAYLOAD = payload
            _LATEST_PAYLOAD['cache_timestamp'] = time.time()
        logger.info("[SIDRCE] Initial cache populated")
    except Exception as e:
        logger.warning(f"[SIDRCE] Initial cache population failed: {e}")
        _LATEST_PAYLOAD.update({'error': 'No telemetry data available', 'fti_avg': 0.0, 'lambda_effective': 0.0})

    # Start background thread
    collector_thread = Thread(target=background_collector, daemon=True, name="SIDRCE-Collector")
    collector_thread.start()
    logger.info("[SIDRCE] Server startup complete")

@app.get("/report")
def report():
    """Return cached metrics report (no I/O per request)"""
    with _CACHE_LOCK:
        if not _LATEST_PAYLOAD:
            return {"error": "Cache not yet initialized", "fti_avg": 0.0, "lambda_effective": 0.0}
        return _LATEST_PAYLOAD.copy()

@app.get("/metrics")
def metrics():
    """Return Prometheus metrics from cache"""
    with _CACHE_LOCK:
        if not _LATEST_PAYLOAD:
            payload = {"fti_avg": 0.0, "lambda_effective": 0.0}
        else:
            payload = _LATEST_PAYLOAD.copy()

    # Prometheus exposition format
    txt = []
    txt.append("# HELP sidrce_fti_avg FinOps Tradeoff Index (avg)")
    txt.append("# TYPE sidrce_fti_avg gauge")
    txt.append(f"sidrce_fti_avg {payload.get('fti_avg', 0.0)}")
    txt.append("# HELP sidrce_lambda_effective Effective lambda")
    txt.append("# TYPE sidrce_lambda_effective gauge")
    txt.append(f"sidrce_lambda_effective {payload.get('lambda_effective', 0.0)}")

    # Optional: cache age metric
    if 'cache_timestamp' in payload:
        age = time.time() - payload['cache_timestamp']
        txt.append("# HELP sidrce_cache_age_seconds Age of cached metrics")
        txt.append("# TYPE sidrce_cache_age_seconds gauge")
        txt.append(f"sidrce_cache_age_seconds {age:.1f}")

    return Response("\n".join(txt)+"\n", media_type="text/plain; version=0.0.4")

@app.get("/health")
def health():
    """Health check endpoint"""
    with _CACHE_LOCK:
        cache_age = time.time() - _LATEST_PAYLOAD.get('cache_timestamp', 0) if _LATEST_PAYLOAD else float('inf')

    healthy = cache_age < (_CACHE_UPDATE_INTERVAL * 3)  # Unhealthy if 3x interval passed

    return {
        "status": "healthy" if healthy else "degraded",
        "cache_age_seconds": cache_age if cache_age != float('inf') else None,
        "cache_interval": _CACHE_UPDATE_INTERVAL
    }

# run: uvicorn sidrce.server:app --host 0.0.0.0 --port 8080
