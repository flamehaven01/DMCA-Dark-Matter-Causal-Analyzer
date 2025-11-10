# -*- coding: utf-8 -*-
"""SIDRCE Server — FastAPI /report & /metrics"""
from __future__ import annotations
import json, os
from fastapi import FastAPI, Response
from .collector import run_collect

app = FastAPI(title="SIDRCE Server", version="1.0.5")

@app.get("/report")
def report():
    payload = run_collect(
        in_path=os.environ.get("SIDRCE_STREAM", "ops_telemetry.jsonl"),
        out_path=os.environ.get("SIDRCE_REPORT", "sidrce_report.json")
    )
    return payload

@app.get("/metrics")
def metrics():
    payload = run_collect(
        in_path=os.environ.get("SIDRCE_STREAM", "ops_telemetry.jsonl"),
        out_path=os.environ.get("SIDRCE_REPORT", "sidrce_report.json")
    )
    # Prometheus exposition
    txt = []
    txt.append("# HELP sidrce_fti_avg FinOps Tradeoff Index (avg)")
    txt.append("# TYPE sidrce_fti_avg gauge")
    txt.append(f"sidrce_fti_avg {payload['fti_avg']}")
    txt.append("# HELP sidrce_lambda_effective Effective lambda")
    txt.append("# TYPE sidrce_lambda_effective gauge")
    txt.append(f"sidrce_lambda_effective {payload['lambda_effective']}")
    return Response("\n".join(txt)+"\n", media_type="text/plain; version=0.0.4")

# run: uvicorn sidrce.server:app --host 0.0.0.0 --port 8080
