# 10-Point Pre-Deployment Checklist — Fine-Tuning

> **"Review like you're the person getting paged at 3am when this breaks."**
>
> Run this checklist before every training run. Run it before every deployment.
> Takes 5 minutes, saves hours of debugging. All 10 must pass.

---

## Layer 1: Data (2 checks)

| # | Check | How to Verify | ❌ If Failed |
|---|-------|--------------|-------------|
| 1 | **All Q&A pairs have `instruction` + `output` fields (no nulls)** | `df[['instruction','output']].isnull().sum()` — both must be 0 | Model trains on garbage. Fix data pipeline. |
| 2 | **No empty strings in `instruction` or `output`** | `(df['instruction'].str.len() == 0).sum()` — must be 0 | Model learns to output nothing. Filter or regenerate. |

## Layer 2: Model (2 checks)

| # | Check | How to Verify | ❌ If Failed |
|---|-------|--------------|-------------|
| 3 | **Model responds to a known-good prompt** | Run `pipe("### Instruction: What is Metformin? ### Response:")` — must return non-empty text | Model is broken or adapter didn't load. Check `PeftModel.from_pretrained()`. |
| 4 | **Response contains expected domain keywords** | `assert "diabetes" in response.lower()` or your domain equivalent | Model didn't learn the domain. Increase epochs or check data quality. |

## Layer 3: Pipeline (4 checks)

| # | Check | How to Verify | ❌ If Failed |
|---|-------|--------------|-------------|
| 5 | **Training completes without OOM** | Monitor `nvidia-smi` during training. Loss curve shows steady decrease. | Reduce batch size, use gradient accumulation, or enable 4-bit quantization. |
| 6 | **Loss decreases (not stuck, not NaN)** | Check training logs: loss should decrease monotonically. `assert not np.isnan(loss)` | NaN → lower learning rate. Flat loss → check data, increase lr. |
| 7 | **Adapter saved correctly (file exists, > 0 bytes)** | `assert os.path.getsize("./adapter/adapter_model.safetensors") > 0` | Training didn't save properly. Check `save_pretrained()` call. |
| 8 | **Saved adapter loads without error** | `PeftModel.from_pretrained(base_model, "./adapter")` — no exceptions | Adapter file is corrupted. Re-save and try again. |

## Layer 4: Production (2 checks)

| # | Check | How to Verify | ❌ If Failed |
|---|-------|--------------|-------------|
| 9 | **No data drift vs baseline on validation set** | Run `scripts/check_drift.py` — drift score < 0.5 | Users asking about topics you never trained on. Collect new data → retrain. |
| 10 | **GPU memory stable during inference** | `nvidia-smi` shows consistent memory usage (no creep). | Memory leak in inference loop. Check `torch.cuda.empty_cache()` or batching. |

---

## Quick Command (Run All)

```bash
# Layer 1: Data
python -m pytest tests/test_data.py -v

# Layer 2: Model
python -m pytest tests/test_model.py -v

# Layer 3: Pipeline
python -m pytest tests/test_pipeline.py -v

# Layer 4: Production
python scripts/check_drift.py
```

---

## Sign-Off

- [ ] All 10 checks passed
- [ ] Training loss curve reviewed
- [ ] Adapter file verified on disk
- [ ] Drift report generated and reviewed
- [ ] Model responses manually spot-checked (3-5 prompts)

**Trained by:** _______ **Date:** _______ **Deployed by:** _______ **Date:** _______
