"""
Layer 4: Production Monitoring — Is It Still Working?
======================================================
Fine-tuned models don't crash — they silently degrade. Monitor continuously.

This script uses Evidently AI to detect data drift between:
  - reference_data: What you trained on (baseline distribution)
  - current_data:   What users are asking now (production distribution)

If drift score exceeds threshold → users are asking about topics you
never trained on → time to retrain!

Usage:
    python scripts/check_drift.py
    python scripts/check_drift.py --reference data/train.csv --current data/live.csv
    python scripts/check_drift.py --threshold 0.3

Exit codes:
    0 = No significant drift (safe to keep serving)
    1 = Drift detected (consider retraining)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Evidently AI — open-source ML monitoring
try:
    from evidently import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Default paths — adjust for your project
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_CURRENT = PROJECT_ROOT / "data" / "live_production_sample.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "drift_report.html"


# ═══════════════════════════════════════════════════════════════
# Sample data generation — for when real production data isn't
# available yet (educational/development use)
# ═══════════════════════════════════════════════════════════════

def generate_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic training (reference) and production (current) data
    for demonstration purposes. In production, replace with real data.

    Training data: Medical Q&A about diabetes/metformin (in-distribution)
    Production data: Mix of medical + new topics (simulates drift)
    """
    # Training data — all about diabetes/endocrinology
    train_data = pd.DataFrame({
        "instruction": [
            "### Instruction: What is Metformin?\n### Response:",
            "### Instruction: How does insulin work?\n### Response:",
            "### Instruction: What is type 2 diabetes?\n### Response:",
            "### Instruction: Normal A1C range?\n### Response:",
            "### Instruction: Metformin side effects?\n### Response:",
            "### Instruction: What is HbA1c?\n### Response:",
            "### Instruction: Diabetes diet plan?\n### Response:",
            "### Instruction: Glucose monitoring?\n### Response:",
            "### Instruction: Metformin dosage?\n### Response:",
            "### Instruction: Insulin resistance treatment?\n### Response:",
        ],
        "output": [
            "Metformin is a biguanide medication for type 2 diabetes.",
            "Insulin is a hormone that allows cells to absorb glucose from blood.",
            "Type 2 diabetes is a condition where cells become resistant to insulin.",
            "Normal A1C is below 5.7%. Prediabetes: 5.7-6.4%. Diabetes: 6.5%+.",
            "Common side effects: nausea, diarrhea, metallic taste, stomach upset.",
            "HbA1c measures average blood sugar over the past 2-3 months.",
            "Focus on fiber-rich foods, lean proteins, and controlled carbohydrates.",
            "Use a glucometer to check blood sugar levels 2-4 times daily.",
            "Starting dose typically 500mg twice daily, max 2550mg per day.",
            "Treatment includes Metformin, lifestyle changes, and sometimes insulin.",
        ],
        "topic": ["diabetes"] * 10,
    })

    # Production data — some medical, some completely new topics (drift!)
    live_data = pd.DataFrame({
        "instruction": [
            "### Instruction: What is Metformin?\n### Response:",           # ✅ In-distribution
            "### Instruction: Diabetes symptoms?\n### Response:",          # ✅ In-distribution
            "### Instruction: How to deploy a React app?\n### Response:",  # ❌ DRIFT: web dev
            "### Instruction: Best Python libraries for ML?\n### Response:", # ❌ DRIFT: ML tools
            "### Instruction: Metformin vs insulin?\n### Response:",       # ✅ In-distribution
            "### Instruction: What is Kubernetes?\n### Response:",         # ❌ DRIFT: DevOps
            "### Instruction: Side effects of Metformin?\n### Response:",  # ✅ In-distribution
            "### Instruction: How to make pasta?\n### Response:",          # ❌ DRIFT: cooking
            "### Instruction: Normal A1C levels?\n### Response:",          # ✅ In-distribution
            "### Instruction: What is Docker?\n### Response:",             # ❌ DRIFT: containers
        ],
        "output": [
            "Metformin is a biguanide medication.",
            "Common symptoms: increased thirst, frequent urination, fatigue.",
            "Use create-react-app or Vite, then deploy to Vercel or Netlify.",
            "Scikit-learn, PyTorch, TensorFlow, HuggingFace transformers.",
            "Metformin is oral medication; insulin is injected. Both lower blood sugar.",
            "Kubernetes orchestrates containerized applications across clusters.",
            "Nausea, diarrhea, stomach upset. Rare: lactic acidosis.",
            "Boil water, cook pasta 8-10 minutes, drain, add sauce.",
            "Normal A1C is below 5.7%. Check every 3-6 months if diabetic.",
            "Docker packages applications into portable containers for consistent deployment.",
        ],
        "topic": [
            "diabetes", "diabetes", "web_dev", "ml_tools",
            "diabetes", "devops", "diabetes", "cooking",
            "diabetes", "devops"
        ],
    })

    return train_data, live_data


# ═══════════════════════════════════════════════════════════════
# Core drift check
# ═══════════════════════════════════════════════════════════════

def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float = 0.5,
    report_path: str = None,
) -> dict:
    """
    Run Evidently data drift report comparing training to production data.

    Args:
        reference_df: Training data (baseline distribution)
        current_df:   Production/live data (current distribution)
        threshold:    Drift score above which we recommend retraining
        report_path:  Where to save the HTML report

    Returns:
        dict with keys: drift_detected (bool), drift_score (float), report_path (str)
    """
    if not EVIDENTLY_AVAILABLE:
        print("[WARN] Evidently AI not installed. Install with: pip install evidently")
        print("   Skipping drift check. Returning no-drift for CI to pass.")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "report_path": report_path or str(DEFAULT_REPORT_PATH),
            "error": "evidently_not_installed",
        }

    print("=" * 60)
    print("Layer 4: Production Drift Monitoring")
    print("=" * 60)
    print(f"  Reference rows: {len(reference_df)}  (training data)")
    print(f"  Current rows:   {len(current_df)}  (production data)")
    print(f"  Drift threshold: {threshold}")
    print()

    # Select text columns for drift analysis (skip instruction format noise)
    text_columns = ["output"]
    # Also check if 'topic' column exists for categorical drift
    if "topic" in reference_df.columns and "topic" in current_df.columns:
        text_columns.append("topic")

    # Run Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[text_columns],
        current_data=current_df[text_columns],
    )

    # Save HTML report
    save_path = report_path or str(DEFAULT_REPORT_PATH)
    report.save_html(save_path)
    print(f"[INFO] Drift report saved to: {save_path}")

    # Parse the drift score from the report
    # Evidently stores metrics in the report's internal dict
    report_dict = report.as_dict()
    drift_score = _extract_drift_score(report_dict)

    drift_detected = drift_score > threshold

    # Print summary
    print()
    print("-" * 60)
    print(f"  Drift Score: {drift_score:.3f}")
    print(f"  Threshold:   {threshold:.3f}")
    if drift_detected:
        print(f"  [FAIL] DRIFT DETECTED - Users asking about topics you never trained on.")
        print(f"     Recommended action: Collect new data -> Retrain model.")
    else:
        print(f"  [PASS] No significant drift - Model is still in-distribution.")
    print("-" * 60)

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "report_path": save_path,
    }


def _extract_drift_score(report_dict: dict) -> float:
    """
    Extract the overall drift score from an Evidently report dict.
    Returns the maximum drift score across all columns.
    """
    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                return float(result.get("drift_share", 0.0))
        # Fallback: try to compute from column-level drift
        for metric in metrics:
            if metric.get("metric") == "DataDriftTable":
                columns = metric.get("result", {}).get("drift_by_columns", {})
                scores = []
                for col_name, col_data in columns.items():
                    scores.append(float(col_data.get("drift_score", 0.0)))
                return max(scores) if scores else 0.0
    except (KeyError, TypeError, ValueError) as e:
        print(f"[WARN] Could not parse drift score: {e}")
    return 0.0


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer 4: Data Drift Detection for Fine-Tuned Models"
    )
    parser.add_argument(
        "--reference", type=str,
        help="Path to reference (training) data CSV"
    )
    parser.add_argument(
        "--current", type=str,
        help="Path to current (production) data CSV"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Drift score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--report", type=str, default=str(DEFAULT_REPORT_PATH),
        help="Path for the HTML drift report"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (no real data files needed)"
    )
    args = parser.parse_args()

    # Load or generate data
    if args.demo or (not args.reference and not os.path.exists(args.reference or "")):
        print("[INFO] Using synthetic demo data (--demo mode)")
        ref_df, cur_df = generate_sample_data()
    else:
        ref_path = args.reference or str(DEFAULT_REFERENCE)
        cur_path = args.current or str(DEFAULT_CURRENT)
        print(f"Loading reference data: {ref_path}")
        ref_df = pd.read_csv(ref_path)
        print(f"Loading current data:   {cur_path}")
        cur_df = pd.read_csv(cur_path)

    # Run drift check
    result = check_drift(
        reference_df=ref_df,
        current_df=cur_df,
        threshold=args.threshold,
        report_path=args.report,
    )

    # Exit code: non-zero if drift detected (for CI/CD gates)
    if result["drift_detected"]:
        print("\n[FAIL] Drift detected. Retraining recommended.")
        sys.exit(1)
    else:
        print("\n[PASS] No drift. Model is healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
