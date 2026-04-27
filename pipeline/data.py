"""
pipeline/data.py
----------------
Stage 1: Data collection and validation.

Design principles:
- Accumulates ALL validation errors before raising (not fail-fast).
  The engineer sees the complete picture in a single run.
- Null-rate check is warning-only: upstream nulls often have legitimate
  causes; silently failing every retrain is worse than logging and continuing.
- Volume and label checks are blocking: training on bad data is worse
  than not training at all.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: Path
    reference_path: Path          # original training data — used for drift checks
    lookback_days: int = 90       # how many days of recent data to train on
    min_samples: int = 500        # refuse to train on fewer than this
    min_positive_rate: float = 0.02   # catch label drift to near-zero
    max_positive_rate: float = 0.40   # catch label drift to near-one
    required_features: Optional[List[str]] = field(default_factory=list)


class DataValidationError(Exception):
    """Raised when data fails validation. Pipeline halts."""
    pass


def load_training_window(config: DataConfig) -> pd.DataFrame:
    """
    Load the most recent `lookback_days` of labeled data.

    Expects parquet files named YYYY-MM-DD.parquet inside config.data_dir.
    In a real system this would query your feature store or data warehouse.
    """
    cutoff = datetime.now() - timedelta(days=config.lookback_days)
    frames = []

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise DataValidationError(
            f"Data directory '{data_dir}' does not exist. "
            "Create it and populate with YYYY-MM-DD.parquet files."
        )

    for path in sorted(data_dir.glob("*.parquet")):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d")
        except ValueError:
            logger.debug("Skipping non-date file: %s", path.name)
            continue
        if file_date >= cutoff:
            frames.append(pd.read_parquet(path))

    if not frames:
        raise DataValidationError(
            f"No data files found in '{data_dir}' "
            f"for the past {config.lookback_days} days. "
            "Widen the lookback window or check the upstream pipeline."
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d samples from %d file(s)", len(df), len(frames))
    return df


def validate_data(df: pd.DataFrame, config: DataConfig) -> None:
    """
    Validates the training window before any model code runs.

    Accumulates all errors before raising DataValidationError so the engineer
    sees the full picture rather than fixing one issue at a time.

    ML-specific checks (not covered by Great Expectations):
      - Minimum sample volume
      - Label column existence and dtype
      - Label rate bounds (catches collection failures and leakage)
      - Required feature presence
      - Null rate (warning-only, non-blocking)
    """
    errors = []

    # ── Volume check ──────────────────────────────────────────────
    if len(df) < config.min_samples:
        errors.append(
            f"Only {len(df)} samples — minimum is {config.min_samples}. "
            "Widen the lookback window or check the upstream pipeline."
        )

    # ── Label checks ──────────────────────────────────────────────
    if "label" not in df.columns:
        errors.append("Column 'label' not found. Schema may have changed.")
    else:
        if df["label"].dtype not in [np.int64, np.int32, np.float64, np.int8, np.uint8]:
            errors.append(
                f"Label column has unexpected dtype: {df['label'].dtype}. "
                "Expected integer or float."
            )
        else:
            positive_rate = float(df["label"].mean())
            if positive_rate < config.min_positive_rate:
                errors.append(
                    f"Positive rate is {positive_rate:.4f} — suspiciously low "
                    f"(minimum: {config.min_positive_rate}). "
                    "Check if labels are being collected correctly."
                )
            if positive_rate > config.max_positive_rate:
                errors.append(
                    f"Positive rate is {positive_rate:.4f} — suspiciously high "
                    f"(maximum: {config.max_positive_rate}). "
                    "Check for label leakage in the upstream pipeline."
                )

    # ── Feature presence ──────────────────────────────────────────
    if config.required_features:
        missing = set(config.required_features) - set(df.columns)
        if missing:
            errors.append(
                f"Missing required features: {sorted(missing)}. "
                "Schema may have changed upstream."
            )

    # ── Null rate (warning, not blocking) ─────────────────────────
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.10]
    if not high_null.empty:
        logger.warning(
            "High null rates detected (>10%%) — will be imputed by preprocessor: %s",
            {col: f"{rate:.1%}" for col, rate in high_null.items()},
        )

    # ── Raise with all errors at once ─────────────────────────────
    if errors:
        raise DataValidationError(
            f"Data validation failed ({len(errors)} error(s)):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        "Data validation passed: %d samples, %.2f%% positive rate",
        len(df),
        float(df["label"].mean()) * 100 if "label" in df.columns else 0,
    )
