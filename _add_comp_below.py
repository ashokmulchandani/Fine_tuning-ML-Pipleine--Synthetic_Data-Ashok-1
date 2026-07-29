with open('finetuning_by_data_type.html', 'r', encoding='utf-8') as f:
    c = f.read()

# Find the closing of the comparison section
marker = "Proprietary (Paid)"
idx = c.find(marker)
if idx < 0:
    print("NOT FOUND")
    exit(1)

# Find the closing backtick after this
close = c.find("`;", idx + 500)
if close < 0:
    print("CLOSE NOT FOUND")
    exit(1)

# The comparison div closes at this point. Add our content BEFORE the closing backtick.
new_content = """</div>

<h3 style="margin-top:1.5rem;">🆚 LLM Fine-Tuning vs Classic ML — Why Different Patterns?</h3>
<div style="overflow-x:auto;">
<table class="tbl">
<thead><tr><th>Pipeline Step</th><th>Classic ML (tabular, scikit-learn)</th><th>Our LLM FT (text, HuggingFace)</th><th>Why Different</th></tr></thead>
<tbody>
<tr style="background:rgba(26,110,52,0.04);"><td><strong>Data Ingestion</strong></td><td>ingest_data.py FACTORY -> Zip/CSV</td><td>format_instruction() -> ONE format: Alpaca</td><td style="font-size:0.7rem;">LLMs read ONE string format. No factory needed.</td></tr>
<tr><td><strong>Missing Values</strong></td><td>STRATEGY: Drop OR Fill (mean)</td><td><strong>Pandera</strong> -> REJECT bad rows</td><td style="font-size:0.7rem;">LLMs learn from text. Nulls corrupt training.</td></tr>
<tr style="background:rgba(26,110,52,0.04);"><td><strong>Feature Engineering</strong></td><td>STRATEGY: Log, Scale, OneHot</td><td>features_to_text() -> Numbers -> English</td><td style="font-size:0.7rem;">LLMs read TEXT, not scaled floats!</td></tr>
<tr><td><strong>Outlier Detection</strong></td><td>STRATEGY: ZScore > 3, IQR</td><td>DBSCAN -> label=-1 (skip noise)</td><td style="font-size:0.7rem;">Clustering auto-flags outliers. No separate step.</td></tr>
<tr style="background:rgba(26,110,52,0.04);"><td><strong>Data Split</strong></td><td>STRATEGY: Train/Test (80/20)</td><td>Train/Val (90/10) — <strong>SAME!</strong></td><td style="font-size:0.7rem;">Only shared pattern between both worlds.</td></tr>
<tr><td><strong>Model Building</strong></td><td>STRATEGY: Swap RF, XGBoost, LR</td><td>TinyLlama + LoRA — <strong>SAME MODEL</strong></td><td style="font-size:0.7rem;">LoRA adapts one model. No swapping needed.</td></tr>
<tr style="background:rgba(26,110,52,0.04);"><td><strong>Model Evaluation</strong></td><td>RMSE=2.3, R2=0.87 (numbers)</td><td><strong>MFT/INV/DIR</strong> behavioral tests</td><td style="font-size:0.7rem;">LLMs need behavior checks, not metrics.</td></tr>
</tbody></table>
</div>

<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;margin:0.75rem 0;font-size:0.65rem;text-align:center;">
  <div style="background:rgba(91,58,168,0.06);border:1px solid rgba(91,58,168,0.2);border-radius:8px;padding:0.5rem;"><strong style="color:var(--violet);">Pandera = DataFrame Validation</strong><br>Like pydantic for pandas. Checks columns, types, nulls.<br><span style="color:var(--tmut);">pip install pandera | Catches 90%% of issues BEFORE training</span></div>
  <div style="background:rgba(0,122,153,0.06);border:1px solid rgba(0,122,153,0.2);border-radius:8px;padding:0.5rem;"><strong style="color:var(--cyan);">MFT/INV/DIR = Behavioral Tests</strong><br>MFT: Knows facts? INV: Robust to phrasing? DIR: Refuses harm?<br><span style="color:var(--tmut);">Phase 6A slide 6.6 | tests/test_model.py | 10 tests</span></div>
</div>
"""

# Insert before the closing `;
c = c[:close] + new_content + c[close:]

bc = c.count('{')
ec = c.count('}')
print(f"Braces: {bc}/{ec} ok={bc==ec}")

with open('finetuning_by_data_type.html', 'w', encoding='utf-8') as f:
    f.write(c)
print("DONE")
