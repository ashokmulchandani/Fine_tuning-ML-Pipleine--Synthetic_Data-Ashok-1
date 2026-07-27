"""
MLflow Tracking — Log Every Fine-Tuning Experiment
====================================================
Drop-in module that records params, metrics, and artifacts for every
training run. Called from train_stage.py (DVC pipeline stage).

Usage:
    from scripts.mlflow_tracking import TrackRun

    with TrackRun("instruction-ft-medical", params) as run:
        # ... train ...
        run.log_metric("final_loss", 0.32)
        run.log_adapter("./adapter_model.safetensors")
        # On exit: auto-logs duration, registers model, prints comparison

Requirements: pip install mlflow
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    import mlflow.transformers
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TrackRun:
    """
    Context manager for MLflow experiment tracking.

    with TrackRun("medical-qa-ft", params) as run:
        trainer.train()
        run.log_metric("final_loss", 0.32)
        run.log_artifact("adapter_model.safetensors")
    """

    def __init__(self, experiment_name: str, params: dict = None,
                 tracking_uri: str = None):
        self.experiment_name = experiment_name
        self.params = params or {}
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"
        )
        self._start_time = None
        self._run = None

    def __enter__(self):
        if not MLFLOW_AVAILABLE:
            print("[MLflow] Not installed. Run: pip install mlflow")
            return self

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self._start_time = time.time()
        self._run = mlflow.start_run(run_name=datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Log all hyperparameters
        mlflow.log_params(self.params)

        # Log environment info
        mlflow.log_param("python_version", os.sys.version.split()[0])

        print(f"\n[MLflow] Experiment: {self.experiment_name}")
        print(f"[MLflow] Run: {self._run.info.run_id[:8]}...")
        print(f"[MLflow] Params: {json.dumps(self.params, indent=2)}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MLFLOW_AVAILABLE or not self._run:
            return

        # Auto-log training duration
        duration = time.time() - self._start_time
        mlflow.log_metric("training_duration_seconds", round(duration, 1))

        # Auto-log GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
                mlflow.log_metric("gpu_memory_mb",
                    torch.cuda.max_memory_allocated(0) / 1024**2)
        except Exception:
            pass

        mlflow.end_run()
        mins = int(duration // 60)
        secs = int(duration % 60)
        print(f"\n[MLflow] Training completed in {mins}m {secs}s")
        print(f"[MLflow] View results: mlflow ui --backend-store-uri {self.tracking_uri}")

    def log_metric(self, key: str, value: float, step: int = None):
        """Log a single metric value."""
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log multiple metrics at once."""
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        """Log a file or directory as an artifact."""
        if MLFLOW_AVAILABLE and self._run:
            mlflow.log_artifact(path)

    def log_adapter(self, adapter_dir: str):
        """Log the LoRA adapter directory + register as model version."""
        if not MLFLOW_AVAILABLE or not self._run:
            return

        # Log adapter files as artifacts
        mlflow.log_artifacts(adapter_dir, "adapter")

        # Register in Model Registry
        try:
            result = mlflow.register_model(
                f"runs:/{self._run.info.run_id}/adapter",
                self.experiment_name
            )
            print(f"[MLflow] Registered model v{result.version} in '{self.experiment_name}'")
            print(f"[MLflow] Stage: None -> Promote with:")
            print(f"  mlflow models transition {self.experiment_name} {result.version} --stage Staging")
        except Exception as e:
            print(f"[MLflow] Model registration skipped: {e}")


def compare_runs(experiment_name: str, top_n: int = 5):
    """
    Print a comparison of the best N runs sorted by final_loss.
    Run from command line: python scripts/mlflow_tracking.py compare
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not installed.")
        return

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        print(f"No experiment named '{experiment_name}'. Run training first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_loss ASC"],
        max_results=top_n,
    )

    print(f"\n{'='*70}")
    print(f"Top {len(runs)} runs for '{experiment_name}' (by final_loss)")
    print(f"{'='*70}")
    print(f"{'Run ID':<10} {'Loss':<8} {'Duration':<10} {'r':<4} {'lr':<10}")
    print("-" * 70)

    for r in runs:
        rid = r.info.run_id[:8]
        loss = r.data.metrics.get("final_loss", "N/A")
        dur = r.data.metrics.get("training_duration_seconds", 0)
        r_val = r.data.params.get("lora_r", "?")
        lr = r.data.params.get("learning_rate", "?")

        dur_str = f"{int(dur//60)}m{int(dur%60)}s" if dur else "N/A"
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        print(f"{rid:<10} {loss_str:<8} {dur_str:<10} {str(r_val):<4} {str(lr):<10}")

    print("-" * 70)
    print(f"\nFull UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")
    return runs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_runs("instruction-ft-medical")
    else:
        # Demo: log a dummy run to verify setup
        with TrackRun("demo-experiment", {"lora_r": 8, "lr": 2e-4}) as run:
            run.log_metric("final_loss", 0.32)
            run.log_metric("eval_accuracy", 0.89)
            print("\nMLflow demo run complete. View with: mlflow ui")
