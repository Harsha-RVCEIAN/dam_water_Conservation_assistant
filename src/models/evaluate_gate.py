"""Model risk and quality gating module for K.R.Sagara Dam inflow forecasting.

Evaluates trained production model artifacts against historical baselines and
justified performance thresholds before authorizing promotion to production.
All thresholds are derived directly from Phase 1 held-out test benchmarks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METADATA_PATH = Path("models/model_metadata.json")
MODEL_ARTIFACT_PATH = Path("models/inflow_forecaster.joblib")

# Justified thresholds derived from Phase 1 held-out test benchmark
MIN_RMSE_IMPROVEMENT_PCT = 5.0    # Random Forest achieved 11.23% over Persistence
MIN_TEST_R2 = 0.50                # Persistence was 0.4796; RF achieved 0.5899
MAX_ACCEPTABLE_WAPE_PCT = 50.0    # RF achieved 45.69% on held-out test set
DESIGN_FRL_FT = 2482.0            # Full Reservoir Level (ft MSL)
HISTORICAL_MAX_INFLOW = 204200.0  # Maximum observed inflow in 36-year record


@dataclass
class QualityGateResult:
    """Detailed summary of model quality and risk checks."""
    passed: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_passed_count": len(self.checks_passed),
            "checks_failed_count": len(self.checks_failed),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "metrics_summary": self.metrics_summary,
        }


def run_model_quality_gate(
    metadata_path: Path = METADATA_PATH,
    model_path: Path = MODEL_ARTIFACT_PATH,
) -> QualityGateResult:
    """Verify that candidate model satisfies all risk and performance criteria.

    Criteria:
    1. Model artifact (.joblib) and metadata (.json) exist and are loadable.
    2. Chronological split ranges are strictly ordered without temporal overlap.
    3. Test RMSE improvement over Persistence baseline is >= 5.0%.
    4. Test R^2 is >= 0.50 (improves upon baseline 0.4796).
    5. Test WAPE is <= 50.0%.
    6. Calibrated 80% prediction intervals exist and have upper > lower.
    7. Domain physical boundaries are properly documented and non-negative.
    """
    passed_checks: List[str] = []
    failed_checks: List[str] = []

    # Check 1: Artifact existence
    if not model_path.exists():
        failed_checks.append(f"Model artifact not found at {model_path}")
        return QualityGateResult(passed=False, checks_failed=failed_checks)
    if not metadata_path.exists():
        failed_checks.append(f"Model metadata not found at {metadata_path}")
        return QualityGateResult(passed=False, checks_failed=failed_checks)

    passed_checks.append("Model artifact and metadata files exist.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Check 2: Chronological Split Integrity
    split = metadata.get("split", {})
    train_end = split.get("train_range", ["", ""])[1]
    val_start = split.get("val_range", ["", ""])[0]
    val_end = split.get("val_range", ["", ""])[1]
    test_start = split.get("test_range", ["", ""])[0]

    if train_end < val_start and val_end < test_start:
        passed_checks.append(f"Chronological split verified (Train end {train_end} < Val start {val_start} < Test start {test_start}).")
    else:
        failed_checks.append(f"Chronological split violation: {train_end} / {val_start} / {test_start}")

    # Check 3: Performance Metrics vs Baselines
    test_metrics = metadata.get("metrics", {}).get("test", {})
    selected = test_metrics.get("selected_model", {})
    persistence = test_metrics.get("persistence", {})
    rmse_improv = test_metrics.get("rmse_improvement_pct", 0.0)

    # 3a. RMSE improvement over persistence baseline
    if rmse_improv >= MIN_RMSE_IMPROVEMENT_PCT:
        passed_checks.append(f"Test RMSE improvement ({rmse_improv:.2f}%) exceeds threshold of {MIN_RMSE_IMPROVEMENT_PCT}%.")
    else:
        failed_checks.append(f"Test RMSE improvement ({rmse_improv:.2f}%) below threshold of {MIN_RMSE_IMPROVEMENT_PCT}%.")

    # 3b. Test R^2 threshold
    r2_val = selected.get("r2", -999.0)
    if r2_val >= MIN_TEST_R2:
        passed_checks.append(f"Test R^2 ({r2_val:.4f}) meets threshold of {MIN_TEST_R2} (Baseline: {persistence.get('r2', 'N/A')}).")
    else:
        failed_checks.append(f"Test R^2 ({r2_val:.4f}) below threshold of {MIN_TEST_R2}.")

    # 3c. Test WAPE threshold
    wape_val = selected.get("wape_pct", 100.0)
    if wape_val <= MAX_ACCEPTABLE_WAPE_PCT:
        passed_checks.append(f"Test WAPE ({wape_val:.2f}%) is within acceptable limit of {MAX_ACCEPTABLE_WAPE_PCT}%.")
    else:
        failed_checks.append(f"Test WAPE ({wape_val:.2f}%) exceeds limit of {MAX_ACCEPTABLE_WAPE_PCT}%.")

    # Check 4: Empirical Prediction Intervals
    intervals = metadata.get("uncertainty_intervals_cusecs", {})
    q10 = intervals.get("residual_quantile_10")
    q90 = intervals.get("residual_quantile_90")
    if q10 is not None and q90 is not None and q90 > q10:
        passed_checks.append(f"Calibrated 80% prediction interval valid: [q10: {q10:.2f}, q90: {q90:.2f}].")
    else:
        failed_checks.append(f"Invalid prediction interval: q10={q10}, q90={q90}")

    # Check 5: Physical Domain Boundaries
    domain = metadata.get("domain_boundaries", {})
    hist_max = domain.get("historical_max_inflow_cusecs", 0.0)
    frl = domain.get("design_full_reservoir_level_ft", 0.0)
    if hist_max >= 200000.0 and frl >= 2480.0:
        passed_checks.append(f"Domain boundaries verified (Historical Max: {hist_max:,.0f} cfs, FRL: {frl:.1f} ft).")
    else:
        failed_checks.append(f"Invalid domain boundaries in metadata: {domain}")

    all_passed = len(failed_checks) == 0

    return QualityGateResult(
        passed=all_passed,
        checks_passed=passed_checks,
        checks_failed=failed_checks,
        metrics_summary={
            "test_rmse_model": selected.get("rmse"),
            "test_rmse_persistence": persistence.get("rmse"),
            "rmse_improvement_pct": rmse_improv,
            "test_r2": r2_val,
            "test_wape_pct": wape_val,
            "empirical_interval_cusecs": [q10, q90],
        },
    )


def main() -> None:
    """CLI runner for model quality gate."""
    result = run_model_quality_gate()
    print("=======================================================")
    print("MODEL QUALITY & RISK GATE EVALUATION REPORT")
    print("=======================================================")
    print(f"Overall Status: {'PASSED [APPROVED FOR PRODUCTION]' if result.passed else 'FAILED [REJECTED]'}\n")
    print("Passed Checks:")
    for c in result.checks_passed:
        print(f"  + {c}")
    if result.checks_failed:
        print("\nFailed Checks:")
        for f in result.checks_failed:
            print(f"  - {f}")
    print("\nMetrics Summary:")
    for k, v in result.metrics_summary.items():
        print(f"  {k}: {v}")
    print("=======================================================")
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
