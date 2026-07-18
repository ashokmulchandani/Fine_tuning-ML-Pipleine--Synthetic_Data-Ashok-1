# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hands-on **LLM fine-tuning learning project** structured as a 10-phase curriculum. Each phase covers theory first, then implementation in Google Colab notebooks. The project progresses from quantization fundamentals through LoRA/QLoRA, instruction tuning, preference alignment (DPO), synthetic data generation, and cloud fine-tuning on Azure AI Foundry, AWS Bedrock, and AWS SageMaker.

This is **not a library or application** — it's a structured learning repository. There is no build, lint, or test suite. Code lives in Colab notebooks (under `notebooks/` and individual phase directories).

## Repo Architecture

- `FINETUNING_PLAN.md` — the master plan: all phases, steps, execution order, formulas, and library references. **Read this first** when starting any phase-related work.
- `FINE_Tuning_Transcripts/` — 6 reference transcripts from YouTube tutorials (Krish Naik, Sunny Savita, Azure, AWS). These are the primary reference material for each phase.
- `Phase_1_Fundamentals/` through `Phase_10_Advanced/` — one directory per phase. Most are currently scaffolds (`.gitkeep` only). As phases are completed, notebooks and scripts land here.
- `notebooks/` — shared Jupyter notebooks.
- `data/` and `models/` — gitignored directories for training datasets and saved adapters/weights.

## Key Libraries (HuggingFace Ecosystem)

The project uses the HuggingFace stack throughout:

| Library | Import/Usage |
|---------|-------------|
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer`, `Trainer`, `TrainingArguments` |
| `peft` | `LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained()`, `prepare_model_for_kbit_training` |
| `trl` | `SFTTrainer`, `DPOTrainer` |
| `bitsandbytes` | `BitsAndBytesConfig` — `load_in_4bit=True`, `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type="nf4"` |
| `datasets` | `load_dataset`, `Dataset.from_dict()` |
| `accelerate` | Multi-GPU distributed training (used implicitly by Trainer) |

## Current Progress

- ✅ Phase 1 (Fundamentals) & Phase 2 (Quantization) — theory complete
- 📖 Phase 3 (LoRA & QLoRA) — theory in progress
- ⬜ Phases 4–10 — not started

## Critical Implementation Patterns

When writing Colab notebooks for this project, follow these conventions from the transcripts:

### LoRA Adapter Merging (Phases 3–6)
When stacking fine-tuning stages (domain adapt → instruction tune → DPO), LoRA adapters must be **merged before the next stage** — do not stack unmerged adapters:
```python
# Load previous adapter, merge, then apply new LoRA
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
model = model.merge_and_unload()
model = get_peft_model(model, new_lora_config)
```

### Quantization Config (Phases 2+)
Standard 4-bit QLoRA setup:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### LoRA Target Modules
For Llama-family models, target `["q_proj", "v_proj"]`. For GPT-style, target `["c_attn"]`. Default rank: `r=8`, `lora_alpha=16`.

### Data Formats
- **Non-instructional** (Phase 4): `{"text": "chunk_content"}` — self-supervised next-token prediction
- **Instruction** (Phase 5): Alpaca format — `### Instruction:\n...\n### Input:\n...\n### Response:\n...`
- **Preference** (Phase 6): `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **Cloud APIs** (Phases 7–9): JSONL with `messages: [{role: "system"}, {role: "user"}, {role: "assistant"}]`

## Transcript-to-Phase Mapping

| Transcript File | Phases Covered |
|----------------|----------------|
| `Fine Tuning LLM Models – Generative AI Course_krish_naik.txt` | 2 (quantization), 3 (LoRA/QLoRA), 4 (Llama 2) |
| `LLM Fine-Tuning Crash Course...` | 1 (pipeline), 4, 5, 6 (DPO) |
| `AI-900 LAB Fine-Tuning GPT Models...` | 7 (Azure, GPT-3.5) |
| `Fine-Tune LLM Models with Ease on Azure AI Foundry.txt` | 7 (Azure, GPT-4o) |
| `Amazon Bedrock – Fine-Tuning Explained...` | 8 (Nova Micro) |
| `Fine-tune Large Language Models on Amazon SageMaker...` | 9 (QLoRA, Spectrum, MLflow) |
