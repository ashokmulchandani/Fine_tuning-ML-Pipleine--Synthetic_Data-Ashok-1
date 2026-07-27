"""
Cost Optimization — Compare Fine-Tuning + Deployment Costs
===========================================================
Estimates training and serving costs across different configurations.
Run before training to pick the cheapest setup that meets your needs.

Usage:
    python scripts/cost_benchmark.py

Output: Cost comparison table + recommendations
"""

# ── Pricing (approximate, July 2026) ─────────────────────────

GPU_HOURLY = {
    "T4 (Colab Free)": 0.00,
    "T4 (AWS g4dn)": 0.53,
    "T4 (GCP n1-standard-4 + T4)": 0.60,
    "A10 (AWS g5)": 0.77,
    "A100 (AWS p4d)": 4.06,
    "A100 (GCP a2-highgpu)": 3.67,
    "H100 (AWS p5)": 14.48,
    "L4 (GCP g2-standard-8)": 0.74,
}

STORAGE_MONTHLY_PER_GB = {
    "S3 Standard": 0.023,
    "GCS Standard": 0.020,
    "Google Drive (free)": 0.00,  # 15 GB free
    "HuggingFace Hub (free)": 0.00,  # Unlimited public
}

SERVING_MONTHLY = {
    "vLLM on T4 (self-hosted)": 387,  # $0.53 x 730 hrs
    "vLLM on A10 (self-hosted)": 562,  # $0.77 x 730 hrs
    "AWS SageMaker (ml.g4dn.xlarge)": 531,  # ~$0.73/hr
    "Azure AI Foundry (Standard)": 1241,  # $1.70/hr
    "Serverless (pay-per-token)": "~$0.50 per 1M tokens",
}

# ── Training Time Estimates (TinyLlama-1.1B, 500 Q&A) ────────

TRAINING_CONFIGS = [
    {"name": "LoRA r=8, INT8, T4 (Colab Free)", "gpu": "T4 (Colab Free)",
     "quant": "8-bit", "rank": 8, "time_min": 4, "cost": 0.00},
    {"name": "LoRA r=8, INT8, T4 (AWS)", "gpu": "T4 (AWS g4dn)",
     "quant": "8-bit", "rank": 8, "time_min": 4, "cost": 0.04},
    {"name": "LoRA r=16, INT8, T4", "gpu": "T4 (AWS g4dn)",
     "quant": "8-bit", "rank": 16, "time_min": 5, "cost": 0.04},
    {"name": "LoRA r=8, FP16, T4 (no quant)", "gpu": "T4 (AWS g4dn)",
     "quant": "none", "rank": 8, "time_min": 8, "cost": 0.07},
    {"name": "LoRA r=8, INT8, A100", "gpu": "A100 (AWS p4d)",
     "quant": "8-bit", "rank": 8, "time_min": 2, "cost": 0.14},
    {"name": "Full FT, A100 (no LoRA)", "gpu": "A100 (AWS p4d)",
     "quant": "none", "rank": "N/A", "time_min": 45, "cost": 3.04},
    {"name": "LoRA r=8, INT4, T4 (QLoRA)", "gpu": "T4 (AWS g4dn)",
     "quant": "4-bit", "rank": 8, "time_min": 5, "cost": 0.04},
]

# ── Savings Techniques ────────────────────────────────────────

SAVINGS = [
    {"technique": "4-bit QLoRA (vs FP16 full FT)", "savings": "~96% GPU VRAM",
     "tradeoff": "~1-2% accuracy loss. Acceptable for most tasks.",
     "phase": "Phase 2 (Quantization)"},
    {"technique": "Gradient Accumulation (x4)", "savings": "~75% VRAM",
     "tradeoff": "4x slower training. Batch size unchanged logically.",
     "phase": "Phase 4 (Non-Instructional)"},
    {"technique": "Gradient Checkpointing", "savings": "~60% VRAM",
     "tradeoff": "20% slower. Recomputed activations instead of storing.",
     "phase": "Phase 3 (LoRA)"},
    {"technique": "Spot/Preemptible Instances", "savings": "60-90% cost",
     "tradeoff": "Can be interrupted. Use checkpointing (save every epoch).",
     "phase": "Phase 6B (MLOps)"},
    {"technique": "KV-Cache Quantization (INT8)", "savings": "~50% serving VRAM",
     "tradeoff": "Minimal quality loss. vLLM supports this natively.",
     "phase": "Phase 6B (vLLM)"},
    {"technique": "Continuous Batching (vLLM)", "savings": "10-100x throughput",
     "tradeoff": "None. Same model, better engine.",
     "phase": "Phase 6B (vLLM)"},
    {"technique": "Serverless (pay-per-token)", "savings": "No idle GPU cost",
     "tradeoff": "Cold start latency. Best for low-traffic apps.",
     "phase": "Phase 6B (Deployment)"},
    {"technique": "DVC Caching (skip unchanged stages)", "savings": "Free GPU time",
     "tradeoff": "None. DVC hashes deps — skips if nothing changed.",
     "phase": "Phase 6B (DVC)"},
]


def print_training_costs():
    """Print training cost comparison table."""
    print("=" * 85)
    print("FINE-TUNING COST COMPARISON")
    print("  Model: TinyLlama-1.1B, Dataset: 500 Q&A pairs (10 epochs)")
    print("=" * 85)
    print(f"{'Configuration':<38} {'GPU':<18} {'Time':<8} {'Cost':<8}")
    print("-" * 85)

    for tc in TRAINING_CONFIGS:
        cost_str = f"${tc['cost']:.2f}" if tc['cost'] > 0 else "FREE"
        print(f"{tc['name']:<38} {tc['gpu']:<18} {tc['time_min']}m{'':<4} {cost_str:<8}")

    print("-" * 85)
    print("\n>> Cheapest option: LoRA r=8 + INT8 on Colab Free T4 (FREE)")
    print(">> Best value:     LoRA r=8 + INT8 on AWS T4 ($0.04, 4 minutes)")
    print(">> NOT worth it:   Full fine-tuning on A100 ($3.04 vs $0.04 for LoRA)")
    print("                    76x more expensive for < 1% accuracy improvement.\n")


def print_serving_costs():
    """Print monthly serving cost comparison."""
    print("=" * 85)
    print("MONTHLY SERVING COST (24/7 operation)")
    print("  Model: TinyLlama-1.1B merged, served with vLLM")
    print("=" * 85)
    print(f"{'Serving Method':<38} {'Monthly Cost':<15} {'Throughput':<15}")
    print("-" * 85)

    for method, cost in SERVING_MONTHLY.items():
        cost_str = f"${cost}/mo" if isinstance(cost, (int, float)) else cost
        throughput = ""
        if "vLLM" in method:
            throughput = "~5000 tok/s"
        elif "SageMaker" in method:
            throughput = "~3000 tok/s"
        elif "Foundry" in method:
            throughput = "~4000 tok/s"
        elif "Serverless" in method:
            throughput = "Per-request"

        print(f"{method:<38} {cost_str:<15} {throughput:<15}")

    print("-" * 85)
    print("\n>> Cheapest:    vLLM on T4 ($387/mo, self-hosted)")
    print(">> Zero-cost:   Serverless if < 10K requests/month")
    print(">> Most expensive: Azure AI Foundry ($1,241/mo)")
    print("                    3.2x more than self-hosted vLLM on same GPU.\n")


def print_savings_techniques():
    """Print cost-saving techniques."""
    print("=" * 85)
    print("COST-SAVING TECHNIQUES (Stackable)")
    print("=" * 85)
    print(f"{'Technique':<35} {'Savings':<15} {'Tradeoff'}")
    print("-" * 85)

    for s in SAVINGS:
        print(f"{s['technique']:<35} {s['savings']:<15} {s['tradeoff'][:50]}")

    print("-" * 85)
    print("\n>> Maximum savings: Stack 4-bit QLoRA + Spot + vLLM + DVC cache")
    print("   -> Training: FREE (Colab) or ~$0.01 (spot T4)")
    print("   -> Serving:  $387/mo (vLLM T4, 24/7) or ~$10/mo (serverless, low-traffic)")
    print("   -> Total ops: < $400/mo for a production fine-tuned LLM\n")


def main():
    print()
    print_training_costs()
    print_serving_costs()
    print_savings_techniques()


if __name__ == "__main__":
    main()
