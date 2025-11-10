# -*- coding: utf-8 -*-
"""SIDRCE Bridge: secure ops telemetry emitter (file JSONL, batch/gzip, AES-GCM)."""
from __future__ import annotations
import base64, gzip, json, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except Exception:
    _HAS_CRYPTO = False

_ENC_KEY_ENV = "SIDRCE_AES_KEY"  # base64-encoded 32 bytes
_DEFAULT_FILE = "ops_telemetry.jsonl"

@dataclass
class EmitterConfig:
    path: str = _DEFAULT_FILE
    enabled: bool = True
    batch_size: int = int(os.environ.get("SIDRCE_BATCH_SIZE", "100"))
    flush_sec: float = float(os.environ.get("SIDRCE_FLUSH_SEC", "3.0"))
    gzip_enabled: bool = os.environ.get("SIDRCE_GZIP", "0").lower() not in ("0", "false")
    roll_bytes: int = int(os.environ.get("SIDRCE_ROLL_BYTES", "10485760"))  # 10MB
    encrypt: bool = os.environ.get(_ENC_KEY_ENV, "").strip() != ""
    windows_safe: bool = True

@dataclass
class TelemetryEmitter:
    cfg: EmitterConfig = field(default_factory=EmitterConfig)

    def __post_init__(self):
        self._buf: List[bytes] = []
        self._t0 = time.time()
        self._key: Optional[bytes] = None
        if self.cfg.encrypt:
            if not _HAS_CRYPTO:
                raise RuntimeError("cryptography not available but encryption enabled")
            try:
                self._key = base64.b64decode(os.environ[_ENC_KEY_ENV])
                if len(self._key) not in (16, 24, 32):
                    raise ValueError("AES key must be 16/24/32 bytes (base64)")
            except Exception as e:
                raise RuntimeError(f"invalid {_ENC_KEY_ENV}: {e}")

    def _maybe_roll(self, p: Path):
        if not p.exists():
            return
        if p.stat().st_size >= self.cfg.roll_bytes:
            idx = int(time.time())
            p.rename(p.with_suffix(p.suffix + f".{idx}.bak"))

    def _enc_envelope(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not self._key:
            return obj
        # AES-GCM per-line, random 12-byte nonce
        import os as _os
        aes = AESGCM(self._key)
        nonce = _os.urandom(12)
        aad = b"dmca.ops.telemetry.v1"
        ct = aes.encrypt(nonce, json.dumps(obj, ensure_ascii=False).encode("utf-8"), aad)
        return {
            "enc": True,
            "alg": "AES-GCM",
            "nonce": base64.b64encode(nonce).decode(),
            "aad": base64.b64encode(aad).decode(),
            "ct": base64.b64encode(ct).decode(),
        }

    def emit_rows(self, rows: Iterable[Dict[str, Any]]):
        if not self.cfg.enabled:
            return
        p = Path(self.cfg.path)
        self._maybe_roll(p)
        mode = "ab" if self.cfg.gzip_enabled else "a"
        fh = gzip.open(p, mode) if self.cfg.gzip_enabled else p.open(mode, encoding="utf-8")
        try:
            for r in rows:
                env = self._enc_envelope(r)
                line = (json.dumps(env, ensure_ascii=False) + "\n")
                line_bytes = line.encode("utf-8") if not self.cfg.gzip_enabled else line.encode("utf-8")
                self._buf.append(line_bytes)
                if len(self._buf) >= self.cfg.batch_size or (time.time() - self._t0) >= self.cfg.flush_sec:
                    for b in self._buf:
                        fh.write(b if isinstance(fh, gzip.GzipFile) else b.decode("utf-8"))
                    self._buf.clear()
                    self._t0 = time.time()
            # final flush
            for b in self._buf:
                fh.write(b if isinstance(fh, gzip.GzipFile) else b.decode("utf-8"))
            self._buf.clear()
        finally:
            fh.close()

# Backward-compatible function
def emit_ops_telemetry(df, stream_path: Optional[str] = None, append: bool = True):
    """Emit ops telemetry from DataFrame to JSONL"""
    path = stream_path or os.environ.get("SIDRCE_STREAM", _DEFAULT_FILE)
    em = TelemetryEmitter(EmitterConfig(path=path))
    cols = [c for c in ("domain","hypothesis","step","intervention","infra_cost_usd","qps","latency_p95_ms") if c in df.columns]
    rows = []
    for _, row in df[cols].iterrows():
        r = {}
        for k in cols:
            v = row[k]
            if isinstance(v, (bool,int)):
                r[k] = int(v)
            elif isinstance(v, float):
                r[k] = float(v)
            else:
                r[k] = v
        rows.append(r)
    em.emit_rows(rows)
