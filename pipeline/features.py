"""
pipeline/features.py
--------------------
Stage 2: Feature engineering — preprocessor fit-and-save.

The training-serving skew guard:
  The preprocessor (imputer + scaler) is fit ONCE on the training window
  and saved as an artifact alongside the model. The serving code loads this
  exact artifact. There is no way to accidentally apply different scaling
  parameters at inference time.

  At each retraining, a NEW preprocessor is fit on the new data window
  and versioned together with the new model. Old preprocessor + old model
  are archived in the rollback slot.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical feature list — must match the serving endpoint's input schema.
# Any change here requires a coordinated update to the serving code.
FEATURE_COLS = [
    "transaction_amount",
    "velocity_1h",
    "velocity_24h",
    "distance_from_home",
    "merchant_category_code",
    "device_fingerprint_match",
    "card_present",
    "hour_of_day",
    "day_of_week",
    "cvv_match",
    "address_match",
    "country_match",
]


def build_preprocessor() -> Pipeline:
    """
    Return an unfitted sklearn Pipeline.

    Imputation before scaling: median imputation handles the upstream null
    rates caught (and warned about) in Stage 1 without blocking the run.
    StandardScaler normalises each feature to zero mean, unit variance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def fit_and_save_preprocessor(
    df: pd.DataFrame,
    artifact_dir: Path,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on the training data and persist the artifact.

    Args:
        df:           Raw training DataFrame (must contain FEATURE_COLS + 'label').
        artifact_dir: Directory to write preprocessor.pkl into.

    Returns:
        (fitted_preprocessor, X_transformed, y)
        X_transformed is ready to pass directly into the training stage.

    Raises:
        KeyError if any feature in FEATURE_COLS is missing from df.
        (Stage 1 validation should have caught this already.)
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise KeyError(
            f"Features missing from DataFrame: {sorted(missing)}. "
            "Stage 1 validation should have caught this."
        )

    X = df[FEATURE_COLS].values
    y = df["label"].values

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    preprocessor_path = artifact_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(
        "Preprocessor fitted on %d samples and saved → %s",
        len(X), preprocessor_path,
    )

    return preprocessor, X_transformed, y


def load_preprocessor(artifact_dir: Path) -> Pipeline:
    """Load the fitted preprocessor from an artifact directory."""
    path = Path(artifact_dir) / "preprocessor.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No preprocessor artifact found at '{path}'. "
            "Run the training pipeline first."
        )
    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor
