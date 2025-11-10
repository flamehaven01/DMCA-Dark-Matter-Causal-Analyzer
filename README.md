# DMCA v1.0.5 — Dark Matter Causal Analyzer

**Risk-hardened ops dashboard & security-enhanced telemetry system with Grafana-based FinOps.**

## ✨ What's New in v1.0.5

- **Security**: AES-GCM optional encryption for `ops_telemetry.jsonl` (per-line or batch mode)
- **I/O**: Batch emitter with size/time flush; Windows-safe encoding & path handling
- **Decoupling reinforced**: DMCA **never** computes FTI/λ; SIDRCE only
- **SIDRCE Server**: Lightweight FastAPI service serving `/report` and `/metrics` (Prometheus)
- **Grafana stack**: Docker Compose with Infinity datasource prewired to SIDRCE
- **Schema**: Extended telemetry contract (supports encrypted envelope)
- **CI**: Matrix for causal libs (present/absent), schema validation, encryption smoke
- **Patch/migration**: Script checks for legacy FTI usage & auto-remediation hints

---

## 🏗️ Architecture

```
DMCA (Analyzer)
  ↓ emits ops telemetry (JSONL)
  ↓
SIDRCE (Certifier/SRE)
  - Collector: Computes FTI/λ_effective
  - Server: FastAPI /report & /metrics
  ↓
Grafana Dashboard
  - Real-time FinOps monitoring
  - Infinity datasource → SIDRCE /report
```

**North Star Metrics (DMCA)**: `causal_effect`, `reliability_score`, `SR9`, `DI2`
**FinOps Metrics (SIDRCE)**: `FTI` (FinOps Tradeoff Index), `λ_effective` (evolution rate)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DMCA-Dark-Matter-Causal-Analyzer.git
cd DMCA-Dark-Matter-Causal-Analyzer

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Generate Sample Telemetry

```bash
# Emit sample telemetry
SIDRCE_EMIT=1 python -c "
import json
with open('ops_telemetry.jsonl', 'w') as f:
    f.write(json.dumps({'domain':'direct_lab','hypothesis':'WIMP','step':1,'intervention':'-','infra_cost_usd':120,'qps':480,'latency_p95_ms':180})+'\\n')
    f.write(json.dumps({'domain':'cosmology','hypothesis':'DarkPhoton','step':1,'intervention':'-','infra_cost_usd':102,'qps':510,'latency_p95_ms':170})+'\\n')
"
```

### 3. Run SIDRCE Collector

```bash
# Compute FTI/λ
python sidrce/collector.py
```

**Output:**
```json
{
  "count": 2,
  "medians": {"qps": 495.0, "latency_p95_ms": 175.0, "infra_cost_usd": 111.0},
  "fti_avg": 1.018,
  "fti_grade_worst": "WARN",
  "lambda_effective": 0.7823
}
```

### 4. Run SIDRCE Server

```bash
# Start FastAPI server
uvicorn sidrce.server:app --host 0.0.0.0 --port 8080

# Test endpoints
curl http://localhost:8080/report
curl http://localhost:8080/metrics  # Prometheus format
```

### 5. Launch Grafana Stack

```bash
# Start services
docker compose up -d

# Open dashboard
# http://localhost:3000 (admin/admin)
# Dashboard auto-provisioned: "DMCA FinOps (SIDRCE)"
```

---

## 🔐 Security: AES-GCM Encryption

### Generate Encryption Key

```bash
python - <<'PY'
import os, base64
print(base64.b64encode(os.urandom(32)).decode())
PY
```

### Enable Encryption

```bash
# .env
SIDRCE_AES_KEY=your_base64_key_here
```

Telemetry will be encrypted per-line with AES-GCM. SIDRCE automatically decrypts when the same key is set.

---

## 📊 Metrics Explained

### FTI (FinOps Tradeoff Index)

```
FTI = normalized_cost / normalized_perf
```

- **FTI ≤ 1.0**: ✅ OK (cost-efficient)
- **FTI ≤ 1.05**: ⚠️ WARN (tune recommended)
- **FTI > 1.05**: ❌ FAIL (gate breach)

### λ_effective (Evolution Rate)

```
λ_effective = λ_SIBG × exp(-λ_ASDPI × days_since_change)
```

- **λ_SIBG**: System learning pressure (from FTI improvement)
- **λ_ASDPI**: Deployment speed damping (risk management)

Higher λ = faster evolution (but more risky)

---

## 🧪 CI/CD Gates

### GitHub Actions Matrix

```yaml
strategy:
  matrix:
    causal_libs: ["none", "full"]
```

- **none**: Tests without causallearn/dowhy (fallback mode)
- **full**: Tests with full causal stack

### Gates Enforced

1. **Schema validation**: All telemetry must match `sidrce/telemetry_contract.json`
2. **FTI gate**: `fti_avg ≤ 1.05` (fails CI if breached)
3. **Encryption smoke**: Verify AES-GCM encrypt/decrypt roundtrip

Run locally:
```bash
python scripts/validate_telemetry.py ops_telemetry.jsonl
python sidrce/collector.py
```

---

## 📁 Project Structure

```
DMCA-Dark-Matter-Causal-Analyzer/
├── src/
│   ├── sidrce_bridge.py      # Telemetry emitter (batch, gzip, AES-GCM)
│   ├── metrics.py             # DEPRECATED (guard stubs)
│   └── causal.py              # Hybrid SCM Fallback
├── sidrce/
│   ├── collector.py           # FTI/λ calculator
│   ├── server.py              # FastAPI /report & /metrics
│   └── telemetry_contract.json
├── grafana/provisioning/
│   ├── datasources/ds.yml
│   └── dashboards/
│       ├── dash.yml
│       └── json/dmca_finops.json
├── config/
│   └── experiment.yaml
├── scripts/
│   └── validate_telemetry.py
├── .github/workflows/
│   └── sidrce-gates.yml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔬 Advanced: PySCF Chemistry Module (Optional)

DMCA v1.0.5 supports optional **PySCF-based crystal form-factor calculations** for DM-electron scattering (WIMP, Axion detection).

### Installation

```bash
pip install pyscf
```

### Example: Silicon Minimal

```python
from src.materials.silicon_pyscf import silicon_minimal
result = silicon_minimal()
print(result)  # {"e_tot": -10.xyz, "q_scan": {...}, "kpts": 8}
```

See [PySCF documentation](https://pyscf.org/user/pbc.html) for advanced usage.

---

## 🐛 Troubleshooting

### "cryptography not available but encryption enabled"

```bash
pip install cryptography>=43.0.0
```

### "Missing libs: causallearn, dowhy"

```bash
pip install causallearn>=0.1.8 dowhy>=0.11
```

Or set `REX_CAUSAL_REQUIRED=0` to use fallback mode.

### Grafana dashboard not loading

Ensure Infinity datasource plugin is installed:
```bash
docker compose logs grafana | grep infinity
```

---

## 📖 Documentation

- [Telemetry Contract](sidrce/telemetry_contract.json)
- [FinOps Specification](docs/finops-spec.md) *(coming soon)*
- [PySCF Integration Guide](docs/pyscf-guide.md) *(coming soon)*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PySCF team for quantum chemistry toolkit
- causallearn & DoWhy projects for causal inference
- Grafana Infinity datasource plugin

---

## 📧 Contact

- **Issues**: https://github.com/yourusername/DMCA/issues
- **Email**: your.email@example.com

---

**Version**: 1.0.5
**Last Updated**: 2025-01-10
