"""
DVC Stage 1: prepare_data — Validate and split training data
=============================================================
Called by: dvc repro prepare_data
Reads raw data, validates schema, splits train/val, writes metrics.

Requirements: pip install pandas pandera pyyaml
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load DVC params
params = yaml.safe_load(open("params.yaml"))

DATASET_PATH = params["prepare"]["dataset_path"]
FORMAT = params["prepare"]["format"]
TEST_SPLIT = params["prepare"]["test_split"]


def validate_schema(df: pd.DataFrame) -> dict:
    """Validate training data against expected schema. Returns metrics dict."""
    required_cols = {"instruction", "output"}
    existing_cols = set(df.columns)

    metrics = {
        "total_rows": len(df),
        "missing_columns": list(required_cols - existing_cols),
        "null_instructions": int(df["instruction"].isnull().sum()) if "instruction" in df else -1,
        "null_outputs": int(df["output"].isnull().sum()) if "output" in df else -1,
        "empty_instructions": int((df["instruction"].str.len() == 0).sum()) if "instruction" in df else -1,
        "empty_outputs": int((df["output"].str.len() == 0).sum()) if "output" in df else -1,
        "valid": False,
    }

    # Check all required columns exist
    if metrics["missing_columns"]:
        print(f"[FAIL] Missing columns: {metrics['missing_columns']}")
        return metrics

    # Check for nulls
    if metrics["null_instructions"] > 0:
        print(f"[FAIL] {metrics['null_instructions']} null instructions found")
        return metrics

    if metrics["null_outputs"] > 0:
        print(f"[FAIL] {metrics['null_outputs']} null outputs found")
        return metrics

    # Check for empty strings
    if metrics["empty_instructions"] > 0:
        print(f"[FAIL] {metrics['empty_instructions']} empty instructions found")
        return metrics

    if metrics["empty_outputs"] > 0:
        print(f"[FAIL] {metrics['empty_outputs']} empty outputs found")
        return metrics

    metrics["valid"] = True
    print(f"[PASS] All {metrics['total_rows']} rows valid")
    return metrics


def main():
    print("=" * 60)
    print("DVC Stage 1: Prepare Data")
    print("=" * 60)

    # Load data
    data_path = Path(DATASET_PATH)
    if not data_path.exists():
        # Generate demo data for initial setup
        print(f"[WARN] {DATASET_PATH} not found. Generating demo data...")
        df = pd.DataFrame({
            "instruction": [
                "### Instruction: What is Metformin?\n### Response:",
                "### Instruction: List side effects of Metformin.\n### Response:",
                "### Instruction: What is the maximum daily dose?\n### Response:",
                "### Instruction: How does Metformin work?\n### Response:",
                "### Instruction: What is type 2 diabetes?\n### Response:",
            ],
            "output": [
                "Metformin is a biguanide medication used to treat type 2 diabetes by reducing hepatic glucose production.",
                "Common side effects include nausea, diarrhea, stomach upset, and metallic taste. Rare: lactic acidosis.",
                "Maximum recommended daily dose is 2,550 mg for adults, divided into 2-3 doses with meals.",
                "Metformin decreases glucose production in the liver and improves peripheral insulin sensitivity.",
                "Type 2 diabetes is a chronic condition where cells become resistant to insulin, causing high blood sugar.",
            ],
        })
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[INFO] Created demo dataset: {data_path} ({len(df)} rows)")

    # Load data — supports BOTH JSON (from notebooks) and CSV
    if str(data_path).endswith('.json'):
        import json
        with open(data_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Ensure required columns exist
        if 'instruction' not in df.columns and 'prompt' in df.columns:
            df['instruction'] = df['prompt']  # DPO format compatibility
        if 'output' not in df.columns and 'chosen' in df.columns:
            df['output'] = df['chosen']       # DPO format compatibility
        # Add input column if missing
        if 'input' not in df.columns:
            df['input'] = ''
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    # Validate
    metrics = validate_schema(df)
    if not metrics["valid"]:
        print("[FAIL] Data validation failed. Fix data and re-run: dvc repro")
        sys.exit(1)

    # Split train/val
    val_size = max(1, int(len(df) * TEST_SPLIT))
    val_df = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)

    train_path = Path("data/train_split.csv")
    val_path = Path("data/val_split.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Val:   {len(val_df)} rows -> {val_path}")

    # Write metrics for DVC
    metrics["train_rows"] = len(train_df)
    metrics["val_rows"] = len(val_df)
    Path("metrics").mkdir(exist_ok=True)
    json.dump(metrics, open("metrics/data_validation.json", "w"), indent=2)

    print("[PASS] Stage 1 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
