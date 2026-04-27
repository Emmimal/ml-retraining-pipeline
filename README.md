# ml-retraining-pipeline
A production-ready ML retraining pipeline with drift detection, validation gates, experiment tracking, and safe deployment with rollback.

# ML Retraining Pipeline

A production-grade ML retraining pipeline with drift-aware triggers, champion/challenger evaluation, and automatic rollback. Part of the *Production ML Engineering* series — Article 03 of 15.

```
Trigger → Data Validation → Feature Engineering → Training → Evaluation Gate → Deploy → Verify → Rollback (if needed)
```

Every stage either passes its exit criteria and moves to the next, or fails loudly and halts. No implicit steps. No tribal knowledge. Runs from a single command, or fires automatically when a drift threshold is breached.

---

## Project Structure

```
ml-retraining-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── triggers.py         # Drift/performance/time-based trigger logic
│   ├── data.py             # Stage 1: Data collection & validation
│   ├── features.py         # Stage 2: Feature engineering & preprocessor
│   ├── train.py            # Stage 3: Multi-candidate training
│   ├── evaluation.py       # Stage 4: Champion/challenger gate
│   ├── deploy.py           # Stage 5: Artifact promotion & rollback
│   ├── orchestrator.py     # Main entry point — wires all 5 stages
│   └── scheduled_flow.py   # Prefect wrapper for scheduled execution
├── tests/
│   └── test_pipeline.py    # 34 tests: stage failures, gate, rollback, smoke
├── data/                   # YYYY-MM-DD.parquet training data files
├── model/
│   ├── artifacts/          # Current production model
│   ├── staging/            # Challenger artifacts during training
│   └── previous/           # Rollback slot (previous production)
├── conftest.py
├── pytest.ini
└── requirements.txt
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite
pytest

# Check triggers — retrain only if drift or performance thresholds are breached
python -m pipeline.orchestrator

# Force retrain regardless of trigger state
python -m pipeline.orchestrator --force

# Manual rollback to the previous model
python -m pipeline.deploy --rollback
```

---

## Pipeline Stages

### Stage 1 — Data Validation

Loads the most recent `lookback_days` of labeled data from date-partitioned parquet files. Accumulates **all** validation errors before raising — the engineer sees the complete picture in a single run.

| Check | Behaviour |
|---|---|
| Sample volume below `min_samples` | Halt |
| `label` column missing or wrong dtype | Halt |
| Positive rate outside [2%, 40%] | Halt |
| Required features missing | Halt |
| Null rate > 10% per feature | Warn and continue |

### Stage 2 — Feature Engineering

Fits a `SimpleImputer(median) → StandardScaler` pipeline on the training window and saves the artifact. The same artifact is loaded by the serving code — there is no way to accidentally apply different scaling parameters at inference time. Model and preprocessor are versioned together and travel together.

### Stage 3 — Training

Trains two candidate architectures (GradientBoostingClassifier, RandomForestClassifier) with 5-fold stratified cross-validation. Selects the winner by validation ROC-AUC. Writes a complete `run_log.json` with full provenance: data window, git commit, all candidate configs, CV results, and the winner's metrics.

### Stage 4 — Evaluation Gate

The challenger must clear two bars before any artifact is promoted:

1. **Absolute minimums** — recall ≥ 0.60, ROC-AUC ≥ 0.80
2. **No regression** — ROC-AUC drop vs current champion ≤ 5%

First deployment uses a relaxed gate (Rule 1 only — no champion to compare against). After that, every retrain is evaluated against the model it would replace.

### Stage 5 — Deployment

Copy-swap promotion with automatic rollback:

1. Archive current production → `model/previous/` (rollback slot)
2. Copy challenger → `model/artifacts/` (production)
3. Poll `/health` until `model_loaded: true` (smoke test)
4. On smoke test failure: restore previous artifacts, return `status: rolled_back`

Rollback is a local file copy — no network dependency, no registry lookup.

---

## Trigger Strategy

Triggers evaluate in priority order. Drift fires first because it is a leading indicator — it predicts performance degradation before metrics move.

```
1. PSI > 0.25 on any feature    → DRIFT       (leading indicator)
2. KS statistic > 0.20          → DRIFT       (leading indicator)
3. Recall < 0.60                → PERFORMANCE (lagging indicator)
4. Cost/1k > $15,000            → PERFORMANCE (lagging indicator)
5. Age > 30 days                → TIME        (failsafe)
```

**PSI thresholds:**

| PSI | Status |
|-----|--------|
| < 0.10 | Stable |
| 0.10 – 0.25 | Warning |
| > 0.25 | Action — retrain |

**Calibrating thresholds:** Look at historical monitoring data and find the PSI value at which recall had already dropped more than 10% from baseline. Set your action threshold just below it. Recalibrate after three months of production data. If starting from scratch, the defaults above are reasonable starting points.

---

## Adding Training Data

The pipeline expects parquet files named `YYYY-MM-DD.parquet` in the `data/` directory:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "transaction_amount":        rng.normal(100, 30, n),
    "velocity_1h":               rng.exponential(2, n),
    "velocity_24h":              rng.exponential(5, n),
    "distance_from_home":        rng.exponential(50, n),
    "merchant_category_code":    rng.integers(1, 20, n).astype(float),
    "device_fingerprint_match":  rng.integers(0, 2, n).astype(float),
    "card_present":              rng.integers(0, 2, n).astype(float),
    "hour_of_day":               rng.integers(0, 24, n).astype(float),
    "day_of_week":               rng.integers(0, 7, n).astype(float),
    "cvv_match":                 rng.integers(0, 2, n).astype(float),
    "address_match":             rng.integers(0, 2, n).astype(float),
    "country_match":             rng.integers(0, 2, n).astype(float),
    "label":                     (rng.random(n) < 0.07).astype(int),
})

Path("data").mkdir(exist_ok=True)
df.to_parquet(f"data/{date.today()}.parquet", index=False)
```

---

## Prefect Scheduling

```bash
pip install prefect

# Run once
python pipeline/scheduled_flow.py

# Deploy as a daily scheduled flow
prefect deploy pipeline/scheduled_flow.py:retraining_flow \
  --name ml-retraining \
  --cron "0 2 * * *"
```

The Prefect wrapper raises `RuntimeError` on pipeline failure so the run is marked FAILED and configured notifications fire. If Prefect is not installed, `scheduled_flow.py` falls back to running the pipeline directly.

---

## Orchestrator Comparison

| | Airflow | Prefect | Kubeflow |
|---|---|---|---|
| Infrastructure overhead | High | Low (cloud) / Medium (self-hosted) | Very high |
| Ease of getting started | Medium | Low | High |
| Kubernetes-native | No | No | Yes |
| ML-specific features | Via providers | Limited | Native |
| Best for | Large orgs, complex dependencies | Teams wanting simplicity | Kubernetes-native ML platforms |

---

## Implementation Notes

| Component | Detail |
|---|---|
| Trigger ordering | Drift → Performance → Time |
| PSI computation | Additive smoothing (1e-9) to avoid `log(0)` on empty bins |
| Data validation | Accumulates all errors before raising; null rate is warning-only |
| Preprocessor | `SimpleImputer(median)` + `StandardScaler` in a single Pipeline artifact |
| Model selection | Two candidates (GBM, RF); winner by validation ROC-AUC |
| Evaluation gate | 5% regression tolerance; relaxed on first deployment |
| Rollback | Copy-swap; local file operation, no network dependency |
| Reproducibility | `random_state=42` throughout |

**Dependencies:**

```
scikit-learn  numpy  pandas  scipy  requests  pyarrow
```

---

## References

- Breck et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness.* IEEE Big Data.
- Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
- Gama et al. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys.
- Yurdakul (2018). *Statistical Properties of Population Stability Index.* Western Michigan University.
- Paleyes et al. (2022). *Challenges in Deploying Machine Learning.* ACM Computing Surveys.

---

## Series

**Production ML Engineering** — 15-part series on building production-grade ML systems.

← Previous: [How to Deploy a Machine Learning Model to Production (Article 02)](https://emitechlogic.com/machine-learning-model-deployment/)  
→ Next: [ML Model Versioning: Track, Roll Back, and Manage Models (Article 04)](https://emitechlogic.com/)
