"""Stage 1 Hands-On: MLflow Tracking — run this, then check localhost:5000"""
from scripts.mlflow_tracking import TrackRun
import time

with TrackRun('hands-on-test', {'lora_r': 8, 'lr': 2e-4, 'epochs': 3}) as run:
    for step in range(3):
        time.sleep(1)  # Fake training — replace with actual trainer.train()
        run.log_metric('loss', 2.0 / (step + 1), step=step)
    run.log_metric('final_loss', 0.65)
    print('Done! Refresh http://localhost:5000')
