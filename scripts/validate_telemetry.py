# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
from jsonschema import validate, Draft7Validator
import logging
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path("sidrce/telemetry_contract.json")
STREAM_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "ops_telemetry.jsonl")

schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
validator = Draft7Validator(schema)

errors = 0
for i, line in enumerate(STREAM_PATH.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip(): continue
    obj = json.loads(line)
    for err in validator.iter_errors(obj):
        errors += 1
        logger.info(f"[schema-error] line={i} path={'.'.join(map(str, err.path))} msg={err.message}")

if errors:
    raise SystemExit(f"schema validation failed: {errors} errors")
logger.info("schema validation: OK")