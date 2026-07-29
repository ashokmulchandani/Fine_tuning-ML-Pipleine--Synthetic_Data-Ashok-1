with open('phase6b_mlops_deployment.html', 'r', encoding='utf-8') as f:
    c = f.read()

# Find TABLE 1 code field end
marker = "All 5 engines: ON by default. You write ZERO code.</strong>'},"
pos = c.find(marker)
if pos < 0:
    # Try alternate
    marker = "ZERO code.</strong>'},"
    pos = c.find(marker)

if pos < 0:
    print("NOT FOUND")
    exit(1)

# Insert a TABLE 3 stage after TABLE 1's substeps
# Find the closing of TABLE 1's substeps array and stages
# Structure: ...]},  <- closes substeps of TABLE 1
#            ]},  <- closes TABLE 1 stages (wait, actually it's different)
# Let me find the end of TABLE 1's substeps closing

# Find the position right after TABLE 1 stage closes
t1_end = c.find("TABLE 2", pos)
if t1_end < 0:
    print("TABLE 2 not found after TABLE 1")
    exit(1)

# Find the closing of TABLE 1 stages:   ]}, {emoji:'🧠',name:'TABLE 2'
# We need to insert a new stage between TABLE 1 and TABLE 2
insert_point = c.rfind("]},", pos, t1_end)
if insert_point < 0:
    print("Insert point not found")
    exit(1)

# New TABLE 3 stage between TABLE 1 and TABLE 2
table3 = """},
    {emoji:'📦',name:'TABLE 3',time:'DVC Pipeline Scripts',desc:'Who wrote what',
     substeps:[
       {n:1,title:'DVC Pipeline — Who Writes Each Script',tag:'dvc',code:'<table style="width:100%;border-collapse:collapse;font-size:0.72rem;font-family:Inter,sans-serif;background:#fff;color:#1a1d23;border:1px solid #dde0e6"><thead><tr style="background:#f3effc"><th style="padding:0.45rem 0.6rem;text-align:left;border-bottom:2px solid #5b3aa8;font-weight:700;color:#5b3aa8">Stage</th><th style="padding:0.45rem 0.6rem;text-align:left;border-bottom:2px solid #5b3aa8;font-weight:700;color:#5b3aa8">Command</th><th style="padding:0.45rem 0.6rem;text-align:left;border-bottom:2px solid #5b3aa8;font-weight:700;color:#5b3aa8">Script (we wrote)</th><th style="padding:0.45rem 0.6rem;text-align:left;border-bottom:2px solid #5b3aa8;font-weight:700;color:#5b3aa8">What It Does</th></tr></thead><tbody><tr style="background:#fff"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#5b3aa8">1. prepare_data</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">python scripts/prepare_data.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/prepare_data.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">Loads CSV, validates schema, splits train/val, writes data_validation.json</td></tr><tr style="background:#f8f9fc"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#5b3aa8">2. train</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">python scripts/train_stage.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/train_stage.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">Loads TinyLlama, applies LoRA, trains with SFTTrainer, logs to MLflow</td></tr><tr style="background:#fff"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#5b3aa8">3. evaluate</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">python scripts/evaluate_stage.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/evaluate_stage.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">Runs behavioral tests (MFT/INV/DIR), Evidently drift check, writes evaluation_metrics.json</td></tr><tr style="background:#f3effc"><td style="padding:0.35rem 0.6rem;font-weight:700;color:#5b3aa8" colspan="4"><strong>Supporting scripts (not in dvc.yaml but used by stages):</strong></td></tr><tr style="background:#f8f9fc"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#b87000">util</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">import from scripts.mlflow_tracking</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/mlflow_tracking.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">TrackRun context manager. Auto-logs params, metrics, adapter, registers model</td></tr><tr style="background:#fff"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#b87000">monitor</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">python scripts/check_drift.py --demo</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/check_drift.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">Standalone drift detection with Evidently AI. GitHub Actions runs this daily</td></tr><tr style="background:#f8f9fc"><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:700;color:#b87000">serve</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-family:monospace;font-size:0.65rem">python scripts/serve_vllm.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-weight:600">scripts/serve_vllm.py</td><td style="padding:0.35rem 0.6rem;border-bottom:1px solid #eef0f5;font-size:0.68rem">Merge LoRA adapter, serve with vLLM on :8000, OpenAI-compatible API</td></tr><tr style="background:#fff"><td style="padding:0.35rem 0.6rem;font-weight:700;color:#b87000">cost</td><td style="padding:0.35rem 0.6rem;font-family:monospace;font-size:0.65rem">python scripts/cost_benchmark.py</td><td style="padding:0.35rem 0.6rem;font-weight:600">scripts/cost_benchmark.py</td><td style="padding:0.35rem 0.6rem;font-size:0.68rem">Training + serving cost comparison. Prints all configs and savings</td></tr></tbody></table>'},
     ]}"""

# Insert between TABLE 1 and TABLE 2
c = c[:insert_point+3] + table3 + c[insert_point+3:]
print(f"Braces: {c.count('{')}/{c.count('}')} ok={c.count('{')==c.count('}')}")
with open('phase6b_mlops_deployment.html', 'w', encoding='utf-8') as f:
    f.write(c)
print("DONE")
