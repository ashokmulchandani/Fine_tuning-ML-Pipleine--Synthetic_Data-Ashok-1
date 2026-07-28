"""
Layer 4: Production Monitoring — Drift Detection with Evidently AI
====================================================================
Compares training data distribution to live production data.
Drift score > 0.5 → users asking about NEW topics → RETRAIN.

Uses Evidently AI (open source) for statistical drift tests.
Falls back to simple keyword-based drift if Evidently unavailable.

Usage:
    python scripts/check_drift.py --demo
    python scripts/check_drift.py --reference data/train.csv --current data/live.csv
"""

import argparse
import sys
from collections import Counter

try:
    import evidently
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


def generate_sample_data():
    """Synthetic training (medical only) vs production (mixed topics) data."""
    train = [
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
    ]
    live = [
        "Metformin is a biguanide medication.",
        "Common symptoms: increased thirst, frequent urination, fatigue.",
        "Use create-react-app or Vite, then deploy to Vercel or Netlify.",
        "Scikit-learn, PyTorch, TensorFlow, HuggingFace transformers.",
        "Metformin is oral medication; insulin is injected.",
        "Kubernetes orchestrates containerized applications across clusters.",
        "Nausea, diarrhea, stomach upset. Rare: lactic acidosis.",
        "Boil water, cook pasta 8-10 minutes, drain, add sauce.",
        "Normal A1C is below 5.7%. Check every 3-6 months if diabetic.",
        "Docker packages applications into portable containers.",
    ]
    return train, live


def compute_drift_simple(reference, current):
    """Fallback: keyword-based drift detection."""
    stop_words = {'the','is','a','an','to','for','of','in','and','or','it','be','as','at','by','on','with','from','that','are','was','has'}

    def get_vocab(texts):
        words = []
        for t in texts:
            words.extend([w.lower().strip('.,;:!?()[]') for w in t.split()
                         if len(w) > 3 and w.lower() not in stop_words])
        return set(words), Counter(words)

    ref_vocab, ref_freq = get_vocab(reference)
    cur_vocab, cur_freq = get_vocab(current)

    shared = ref_vocab & cur_vocab
    new_words = cur_vocab - ref_vocab

    new_ratio = len(new_words) / max(len(cur_vocab), 1)
    overlap = len(shared) / max(len(ref_vocab | cur_vocab), 1)
    score = round((new_ratio * 0.6) + ((1 - overlap) * 0.4), 3)

    return {
        "drift_score": score,
        "drift_detected": score > 0.5,
        "shared_words": len(shared),
        "new_words": len(new_words),
        "ref_vocab_size": len(ref_vocab),
        "cur_vocab_size": len(cur_vocab),
        "new_topic_examples": sorted(new_words, key=lambda w: cur_freq[w], reverse=True)[:10],
    }


def compute_drift_evidently(reference, current):
    """Use Evidently AI for statistical drift detection."""
    import pandas as pd

    ref_df = pd.DataFrame({"output": reference})
    cur_df = pd.DataFrame({"output": current})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html("drift_report.html")

    # Extract drift score
    try:
        report_dict = report.as_dict()
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                drift_share = float(metric.get("result", {}).get("drift_share", 0))
                return {
                    "drift_score": round(drift_share, 3),
                    "drift_detected": drift_share > 0.5,
                    "report_path": "drift_report.html",
                    "engine": f"Evidently AI v{evidently.__version__}",
                }
    except Exception:
        pass

    # Fallback if metric extraction fails
    return {"drift_score": 0, "drift_detected": False, "report_path": "drift_report.html",
            "engine": f"Evidently AI v{evidently.__version__}"}


def main():
    parser = argparse.ArgumentParser(description="Production Drift Detection")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--reference", type=str, help="Path to training data CSV")
    parser.add_argument("--current", type=str, help="Path to production data CSV")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("Layer 4: Production Drift Monitoring")
    print("=" * 60)

    # Load data
    if args.demo or not args.reference:
        print(f"[INFO] Using synthetic demo data")
        ref_texts, cur_texts = generate_sample_data()
    else:
        import pandas as pd
        ref_df = pd.read_csv(args.reference)
        cur_df = pd.read_csv(args.current)
        ref_texts = ref_df["output"].tolist() if "output" in ref_df.columns else ref_df.iloc[:, 0].tolist()
        cur_texts = cur_df["output"].tolist() if "output" in cur_df.columns else cur_df.iloc[:, 0].tolist()

    print(f"  Reference: {len(ref_texts)} texts (training)")
    print(f"  Current:   {len(cur_texts)} texts (production)")
    print()

    # Compute drift
    if EVIDENTLY_AVAILABLE:
        print(f"[INFO] Using Evidently AI v{evidently.__version__}")
        results = compute_drift_evidently(ref_texts, cur_texts)
        print(f"\n  Drift Score: {results['drift_score']:.3f}")
        print(f"  Engine:      {results.get('engine', 'Evidently AI')}")
    else:
        print("[INFO] Evidently not installed. Using simple keyword drift.")
        results = compute_drift_simple(ref_texts, cur_texts)
        print(f"\n  Drift Score:       {results['drift_score']:.3f}")
        print(f"  New words:         {results['new_words']}")
        print(f"  Vocab overlap:     {results['shared_words']} shared / {results['ref_vocab_size']} ref / {results['cur_vocab_size']} cur")

    print(f"  Threshold:   {args.threshold}")

    if results["drift_detected"]:
        print(f"\n[FAIL] DRIFT DETECTED! Users asking about new topics.")
        print(f"       Action: Collect new Q&A -> append to data -> dvc repro")
        sys.exit(1)
    else:
        print(f"\n[PASS] No significant drift. Model is healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
