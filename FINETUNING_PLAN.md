# LLM FINE-TUNING MASTER PLAN

## Overview

A complete fine-tuning learning plan covering theory, hands-on practice across multiple platforms (local, Azure AI Foundry, AWS Bedrock, AWS SageMaker), and production deployment.

---

## Phase 1: Fine-Tuning Fundamentals (Theory)

> Understand WHY and WHAT before HOW.

| Step | Task |
|------|------|
| 1.1 | Understand the 3-stage LLM training pipeline: Unsupervised Pre-training → SFT → Preference Alignment |
| 1.2 | Pre-training vs Fine-tuning — "general knowledge" vs "specialized skill" |
| 1.3 | Full fine-tuning vs Partial fine-tuning — when to use each |
| 1.4 | Parameter Efficient Fine-Tuning (PEFT) — why we don't train all 7B+ parameters |
| 1.5 | Fine-tuning vs RAG — when to choose which (cost, data freshness, accuracy tradeoffs) |
| 1.6 | Data types: Non-instructional (plain text/PDF) vs Instructional (Q&A pairs) vs Preference (chosen/rejected) |

## Phase 2: Quantization (Theory + Math)

> How to fit large models on small GPUs.

| Step | Task |
|------|------|
| 2.1 | What is quantization — FP32 → FP16 → INT8 → INT4 (higher memory → lower memory) |
| 2.2 | Full Precision (FP32) vs Half Precision (FP16) — how numbers are stored (sign + exponent + mantissa) |
| 2.3 | Symmetric quantization — scale factor formula: `(x_max - x_min) / (q_max - q_min)` |
| 2.4 | Asymmetric quantization — zero point calculation for non-centered distributions |
| 2.5 | Post-Training Quantization (PTQ) — quantize pre-trained model, some accuracy loss |
| 2.6 | Quantization-Aware Training (QAT) — quantize then fine-tune with new data to recover accuracy |
| 2.7 | BitsAndBytes library — `load_in_4bit=True`, NF4 quantization type, compute dtype BF16 |
| 2.8 | 1-bit LLMs (BitNet) — ternary weights {-1, 0, 1}, addition-only (no multiplication), Pareto improvement |

## Phase 3: LoRA & QLoRA (Theory + Math)

> The core technique for efficient fine-tuning.

| Step | Task |
|------|------|
| 3.1 | Full parameter fine-tuning problem — 175B params = massive GPU/RAM requirement |
| 3.2 | LoRA concept — track weight CHANGES (ΔW), not update all weights |
| 3.3 | Matrix decomposition — decompose ΔW (d×d) into B (d×r) × A (r×d), where r << d |
| 3.4 | Rank selection — r=1,2,4,8,16,64; higher rank = more params but still << full model |
| 3.5 | LoRA equation: `W_new = W_0 + B×A` (pre-trained weights + decomposed tracked changes) |
| 3.6 | Parameter savings — 175B model with r=8 → only 37.7M trainable params (0.02%) |
| 3.7 | QLoRA — quantize model to 4-bit THEN apply LoRA; fits on single GPU |
| 3.8 | LoRA config params: `r`, `lora_alpha`, `lora_dropout`, `target_modules` (q_proj, v_proj) |
| 3.9 | When to use high rank — complex tasks, domain shift; low rank — simple adaptation |
| 3.10 | LoRA adapter merging — delta patch must be merged before next training stage (no stacking!) |

## Phase 4: Non-Instructional Fine-Tuning (Domain Adaptation)

> Teach model your domain language from plain text/PDFs.

| Step | Task | Platform |
|------|------|----------|
| 4.1 | Load PDF data with PyMuPDF, extract text | Colab |
| 4.2 | Chunk text — paragraph splitting, regex patterns, respect model context window | Colab |
| 4.3 | Create HuggingFace Dataset format — `{"text": "chunk1"}, {"text": "chunk2"}` | Colab |
| 4.4 | Tokenize with AutoTokenizer — truncation, padding, max_length=512 | Colab |
| 4.5 | Set labels = input_ids.copy() (self-supervised next-token prediction) | Colab |
| 4.6 | Load base model (TinyLlama/Llama) with BitsAndBytes 8-bit quantization | Colab |
| 4.7 | Configure LoRA — r=8, alpha=16, target_modules=["q_proj","v_proj"] | Colab |
| 4.8 | Define TrainingArguments — epochs, batch_size, learning_rate, save_steps | Colab |
| 4.9 | Train with HuggingFace Trainer — `trainer.train()` | Colab |
| 4.10 | Test: give partial sentence → model completes with domain-specific text | Colab |

## Phase 5: Instruction Fine-Tuning (SFT)

> Teach model to follow instructions and generate structured answers.

| Step | Task | Platform |
|------|------|----------|
| 5.1 | Understand instruction data formats: Alpaca (instruction/input/output), ChatML (system/user/assistant) | Theory |
| 5.2 | Prepare instruction dataset — CSV/JSON with instruction + input + output columns | Colab |
| 5.3 | Format as single string: `### Instruction: {q}\n### Input: {ctx}\n### Response: {ans}` | Colab |
| 5.4 | Synthetic data generation — use GPT-4o to create Q&A from domain PDFs | Colab |
| 5.4a | Synthetic data with **Gretel AI** — generate privacy-safe tabular + text datasets for fine-tuning (PII-free training data) | Gretel Cloud |
| 5.4b | Synthetic data with **Mostly AI** — generate statistically faithful synthetic tabular data (financial/CRM records) for instruction datasets | Mostly AI Cloud |
| 5.4c | Synthetic data with **Tonic AI** — clone production DB to synthetic dev/test version preserving relational integrity for pipeline testing | Tonic Cloud |
| 5.5 | Load previously domain-adapted model (from Phase 4) | Colab |
| 5.6 | Apply LoRA correctly — use `PeftModel.from_pretrained()` to load existing adapter, merge, then `get_peft_model()` for new LoRA | Colab |
| 5.7 | Train with SFTTrainer (from TRL library) | Colab |
| 5.8 | Optional: mask labels before response token (model only learns from response portion) | Colab |
| 5.9 | Compare outputs: base model vs domain-adapted vs instruction-tuned | Colab |
| 5.10 | Push model to HuggingFace Hub | Colab |

## Phase 5b: Synthetic Data Generation for Fine-Tuning (Privacy-Safe Training Data)

> Generate realistic but fake training data — critical for regulated environments (finance, health, government).

| Step | Task | Platform |
|------|------|----------|
| 5b.1 | Understand WHY synthetic data — GDPR/HIPAA/Privacy Act compliance, no real PII in training sets | Theory |
| 5b.2 | Gretel AI — install `gretel-client`, connect API, load real CSV, generate synthetic version | Gretel Cloud / Colab |
| 5b.3 | Gretel — evaluate quality: `gretel evaluate` — SQS score (Statistical Quality Score), correlation heatmaps | Colab |
| 5b.4 | Gretel — generate synthetic text data (call transcripts, support tickets) for LLM fine-tuning | Colab |
| 5b.5 | Mostly AI — upload tabular dataset (transactions, CRM records), configure privacy settings, download synthetic output | Mostly AI Cloud |
| 5b.6 | Mostly AI — compare real vs synthetic distributions (histograms, correlation matrices) | Mostly AI Cloud |
| 5b.7 | Tonic AI — connect to Postgres/Supabase, generate subset with referential integrity preserved for dev/test | Tonic Cloud |
| 5b.8 | Convert synthetic tabular data → instruction format (row → Q&A pair) for fine-tuning pipeline | Colab |
| 5b.9 | Fine-tune model on synthetic dataset, compare loss curves vs real data fine-tune | Colab |
| 5b.10 | Use case: generate 10,000 synthetic call transcripts → fine-tune Whisper/LLM for call centre domain | Colab |

## Phase 6: Preference Alignment (DPO/RLHF)

> Align model responses with human preferences — safe, helpful, honest.

| Step | Task | Platform |
|------|------|----------|
| 6.1 | Understand RLHF — reward model trained on human rankings, PPO algorithm | Theory |
| 6.2 | Understand DPO — direct preference optimization, no reward model needed, supervised learning | Theory |
| 6.3 | DPO loss formula: `L = -E[σ(β × (log(π_θ(y+|x)/π_ref(y+|x)) - log(π_θ(y-|x)/π_ref(y-|x))))]` | Theory |
| 6.4 | Prepare preference data — columns: `prompt`, `chosen`, `rejected` | Colab |
| 6.5 | Load instruction-tuned model, merge LoRA adapter properly | Colab |
| 6.6 | Configure DPOTrainer from TRL — beta=0.1-0.5, reference model = instruction model | Colab |
| 6.7 | Train with DPO — model learns to prefer chosen over rejected responses | Colab |
| 6.8 | Compare: instruction model vs preference-aligned model (tone, safety, helpfulness) | Colab |
| 6.9 | Understand RLAIF — AI-generated feedback for scalability (Anthropic technique) | Theory |

## Phase 6B: MLOps & Deployment

> Track, version, deploy, and monitor your fine-tuned models — the operational side.

| Step | Task | Platform |
|------|------|----------|
| 6B.1 | MLflow experiment tracking — log params, metrics, and artifacts for every run | Local/Cloud |
| 6B.2 | Data versioning with DVC — track which dataset produced which model | Local |
| 6B.3 | Model Registry — version models, promote through stages (Staging → Production) | MLflow |
| 6B.4 | Compare runs — find the best hyperparameters across experiments | MLflow UI |
| 6B.5 | CI/CD pipelines — auto-trigger retraining on new data (GitHub Actions) | Cloud |
| 6B.6 | Deployment patterns — batch inference vs real-time API vs streaming | Theory |
| 6B.7 | Production serving with vLLM — 10× faster than HuggingFace pipeline | Local/Cloud |
| 6B.8 | Monitoring & drift detection — when to retrain | Production |
| 6B.9 | Cost optimization — spot instances, quantization, caching, 20× cheaper | Cloud |

## Phase 7: Azure AI Foundry Fine-Tuning (Cloud - No Code)

> Fine-tune GPT models on Azure without writing code.

| Step | Task | Platform |
|------|------|----------|
| 7.1 | Create Azure AI Foundry project (East US 2 region for fine-tuning support) | Azure Portal |
| 7.2 | Deploy base GPT-4o model, test in playground with system prompt | Azure AI Foundry |
| 7.3 | Prepare training data in JSONL format (messages: system/user/assistant) | Local |
| 7.4 | Upload training data — direct upload or Azure Blob with SAS token | Azure AI Foundry |
| 7.5 | Configure fine-tuning job — model suffix, epochs, batch size, learning rate | Azure AI Foundry |
| 7.6 | Monitor training — loss curves, token accuracy metrics | Azure AI Foundry |
| 7.7 | Deploy fine-tuned model (Standard deployment, reduce TPM for cost) | Azure AI Foundry |
| 7.8 | Test in playground — compare base vs fine-tuned responses | Azure AI Foundry |
| 7.9 | Understand costs — training (cheap per 1K tokens), hosting ($1.70/hr!), inference (per token) | Azure Pricing |
| 7.10 | Cleanup — delete deployment immediately after testing to avoid hosting charges | Azure AI Foundry |

## Phase 8: AWS Bedrock Fine-Tuning (Cloud - Console)

> Fine-tune foundation models on AWS Bedrock.

| Step | Task | Platform |
|------|------|----------|
| 8.1 | Prepare JSONL dataset — system/user/assistant format for Amazon Nova/Titan | Local |
| 8.2 | Upload dataset to S3 bucket | AWS Console |
| 8.3 | Create fine-tuning job in Bedrock — select model (Nova Micro), set hyperparameters | AWS Console |
| 8.4 | Hyperparameters: epochs (1-5), batch size, learning rate, warmup steps | AWS Console |
| 8.5 | Configure IAM role for S3 read/write access | AWS Console |
| 8.6 | Monitor job (~40-60 min for small datasets) | AWS Console |
| 8.7 | Setup inference — on-demand deployment vs provisioned throughput | AWS Console |
| 8.8 | Test in Bedrock playground — chat mode with custom model | AWS Console |
| 8.9 | Review S3 validation results | AWS Console |
| 8.10 | Cleanup — delete deployment, custom model, S3 bucket, IAM role | AWS Console |

## Phase 9: AWS SageMaker Fine-Tuning (Cloud - Code)

> Production-grade fine-tuning with full control on SageMaker.

| Step | Task | Platform |
|------|------|----------|
| 9.1 | Understand SageMaker training jobs — ephemeral compute, pay-per-use | Theory |
| 9.2 | Prepare data in messages format, upload to S3 | SageMaker Studio |
| 9.3 | QLoRA fine-tuning — write training script with accelerate for multi-GPU | SageMaker |
| 9.4 | Spectrum fine-tuning — signal-to-noise analysis to find high-impact layers | SageMaker |
| 9.5 | Configure recipes YAML — model, max_seq_length, epochs, batch_size, precision | SageMaker |
| 9.6 | Kick off training job with HuggingFace estimator or custom container | SageMaker |
| 9.7 | MLflow integration — log metrics, loss curves, hyperparameters automatically | SageMaker |
| 9.8 | Compare QLoRA vs Spectrum — overlay loss curves in MLflow, convergence speed | SageMaker |
| 9.9 | Evaluate: ground truth validation, deterministic Q&A accuracy | SageMaker |

## Phase 10: Advanced Techniques & Production

> From Phase 10 of CUDA plan + additional production patterns.

| Step | Task | Platform |
|------|------|----------|
| 10.1 | Fine-tune YOLOv8 on custom object detection dataset | Colab |
| 10.2 | Fine-tune VGG16/ResNet on Chicken Disease dataset (transfer learning) | Colab |
| 10.3 | Data augmentation for small datasets — flip, rotate, color jitter | Colab |
| 10.4 | Evaluate vision models: precision, recall, mAP, confusion matrix | Colab |
| 10.5 | Export fine-tuned model to TensorRT for deployment | Colab |
| 10.6 | Fine-tune GPT-2/Llama on custom text data with LoRA + PEFT | Colab |
| 10.7 | Prepare datasets — instruction format, chat format, preference format | Colab |
| 10.8 | Deploy fine-tuned LLM with vLLM or TGI (Text Generation Inference) | Local/Cloud |
| 10.9 | Compare: base model vs fine-tuned on domain-specific tasks | Colab |
| 10.10 | DoRA (Weight Decomposition Low-Rank Adaptation) — improved LoRA variant | Colab |
| 10.11 | IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) | Colab |
| 10.12 | Adapter Layers — append small layers to transformer blocks | Colab |
| 10.13 | Prefix Tuning & Prompt Tuning — soft prompt methods | Colab |
| 10.14 | Multi-framework comparison: HuggingFace TRL vs Axolotl vs Unsloth vs LLaMA Factory | Colab |

---

## Execution Order (Recommended)

| Session | What to Complete | Time |
|---------|-----------------|------|
| Session 1 | Phase 1: Fine-tuning fundamentals theory (1.1–1.6) | 1-2 hrs |
| Session 2 | Phase 2: Quantization theory + math (2.1–2.8) | 2-3 hrs |
| Session 3 | Phase 3: LoRA & QLoRA theory + math (3.1–3.10) | 2-3 hrs |
| Session 4 | Phase 4: Non-instructional fine-tuning on PDF (4.1–4.10) | 3-4 hrs |
| Session 5 | Phase 5: Instruction fine-tuning with SFT (5.1–5.10) | 3-4 hrs |
| Session 6 | Phase 5b: Synthetic data generation — Gretel, Mostly AI, Tonic (5b.1–5b.10) | 2-3 hrs |
| Session 7 | Phase 6: DPO preference alignment (6.1–6.9) | 3-4 hrs |
| Session 7b | Phase 6B: MLOps — MLflow, DVC, vLLM, monitoring (6B.1–6B.9) | 2-3 hrs |
| Session 8 | Phase 7: Azure AI Foundry fine-tuning (7.1–7.10) | 2-3 hrs |
| Session 9 | Phase 8: AWS Bedrock fine-tuning (8.1–8.10) | 2-3 hrs |
| Session 10 | Phase 9: AWS SageMaker fine-tuning (9.1–9.9) | 3-4 hrs |
| Session 11 | Phase 10: Vision model fine-tuning (10.1–10.5) | 3-4 hrs |
| Session 12 | Phase 10: LLM fine-tuning + advanced PEFT (10.6–10.14) | 3-4 hrs |

---

## Transcript Coverage Map

| Transcript | Covers |
|------------|--------|
| **Krish Naik - Fine Tuning LLM Models** | Phase 2 (quantization math), Phase 3 (LoRA/QLoRA math), Phase 4 (Llama 2 fine-tuning), 1-bit LLMs, Google Gemma fine-tuning |
| **LLM Fine-Tuning Crash Course (Sunny)** | Phase 1 (LLM training pipeline), Phase 4 (non-instructional on PDF), Phase 5 (instruction FT), Phase 6 (DPO/RLHF), LoRA adapter merging |
| **Azure AI Foundry - Step by Step** | Phase 7 (GPT-3.5 Turbo fine-tuning, SAS tokens, metrics, deployment) |
| **Azure AI Foundry - Kirk** | Phase 7 (GPT-4o fine-tuning, cost breakdown, JSONL format, playground testing) |
| **Amazon Bedrock Fine-Tuning** | Phase 8 (Nova Micro, hyperparameters, S3 upload, on-demand inference) |
| **AWS SageMaker Fine-Tuning** | Phase 9 (QLoRA vs Spectrum, MLflow, training jobs, recipes, multi-GPU) |

---

## Key Libraries & Tools

| Library | Purpose |
|---------|---------|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA, QLoRA, DoRA, IA3, adapter configs |
| `trl` | SFTTrainer, DPOTrainer, PPOTrainer |
| `bitsandbytes` | 4-bit/8-bit quantization |
| `datasets` | HuggingFace dataset loading/formatting |
| `accelerate` | Multi-GPU distributed training |
| `mlflow` | Experiment tracking, metrics logging |
| `PyMuPDF (fitz)` | PDF text extraction |
| `gretel-client` | Synthetic data generation (tabular + text) |
| `mostlyai` | Synthetic tabular data generation |
| `tonic` | Production DB cloning for dev/test |
| `vLLM` / `TGI` | Production LLM serving |
| `Axolotl` / `Unsloth` / `LLaMA Factory` | Alternative fine-tuning frameworks |

---

## Key Formulas

### Quantization Scale Factor
```
scale = (x_max - x_min) / (q_max - q_min)
quantized_value = round(original_value / scale)
```

### LoRA
```
W_new = W_0 + B × A
where W_0 = frozen pre-trained weights (d × d)
      B = trainable matrix (d × r)
      A = trainable matrix (r × d)
      r = rank (hyperparameter, typically 8)
```

### DPO Loss
```
L_DPO = -E[log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

where π_θ = policy model (being trained)
      π_ref = reference model (frozen instruction model)
      y_w = chosen/winning response
      y_l = rejected/losing response
      β = temperature (0.1-0.5)
```

---

## GitHub References

| Resource | URL |
|----------|-----|
| Krish Naik Fine-tuning | `https://github.com/krishnaik06/Finetuning-LLM.git` |
| Sunny Complete LLM Fine-tuning | `https://github.com/sunnysavita10/Complete-LLM-Finetuning.git` |
| HuggingFace PEFT | `https://github.com/huggingface/peft` |
| HuggingFace TRL | `https://github.com/huggingface/trl` |
| Spectrum (SNR-based layer selection) | Search GitHub for `spectrum fine-tuning` |

---

## Interactive Learning Modules

Click any link to open the interactive HTML in your browser. (Hosted via GitHub Pages)

**Base URL:** `https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/`

### 📊 Full Dashboard
- [finetuning_plan_dashboard.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/finetuning_plan_dashboard.html) — Complete dashboard with all phases, progress bar, formulas, and session timeline

### 🎓 Phase 1: Fine-Tuning Fundamentals
- [phase1_fundamentals.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase1_fundamentals.html) — 6 interactive slides covering the 3-stage pipeline, pre-training vs fine-tuning, PEFT, FT vs RAG, and data types

### ⚙️ Phase 2: Quantization
- [phase2_quantization.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase2_quantization.html) — 8 interactive slides covering FP32→INT4, symmetric vs asymmetric, PTQ, QAT, BitsAndBytes, and BitNet
- [bitnet_1_58bit_deep_dive.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/bitnet_1_58bit_deep_dive.html) — 9-slide deep dive into 1.58-bit LLMs (ternary weights, log₂(3), Pareto improvement)

### 🧬 Phase 3: LoRA & QLoRA (Theory)
- [phase3_lora_qlora.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase3_lora_qlora.html) — 10 interactive slides covering LoRA concept, matrix decomposition, rank, equation, QLoRA, config, adapter merging
- [lora_target_modules_explained.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_target_modules_explained.html) — Visual explainer of Q/K/V/O in attention + library analogy
- [lora_adapters_per_layer.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_adapters_per_layer.html) — Visual showing 32 layers × independent adapters per module
- [model_memory_lora_calculator.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/model_memory_lora_calculator.html) — Interactive calculator: FP32→quantization→LoRA. Change rank, precision, target modules — everything updates live

### 🤖 Phase 5b: Synthetic Data Generation
- [phase5b_synthetic_data.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase5b_synthetic_data.html) — 10 slides: Gretel AI, Mostly AI, Tonic AI, synthetic → instruction format

### 🔧 Phase 6B: MLOps & Deployment
- [phase6b_mlops_deployment.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6b_mlops_deployment.html) — 9 slides: MLflow, DVC, model registry, CI/CD, vLLM, monitoring, cost optimization

### ⚖️ Phase 6: Preference Alignment (DPO)
- [phase6_dpo_preference.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6_dpo_preference.html) — 9 slides + 5-cell Colab notebook
- [dpo_loss_simplified.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/dpo_loss_simplified.html) — DPO loss explained: teacher analogy + beta volume knob

### 💬 Phase 5: Instruction Fine-Tuning (SFT)
- [phase5_instruction_finetune.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase5_instruction_finetune.html) — 10 interactive slides + Notebook with 6 Colab cells
- [instruction_format_visual.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/instruction_format_visual.html) — Visual: how Q&A JSON becomes training strings
- [training_loss_explained.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/training_loss_explained.html) — What training loss means (with loss curve)

### 🔬 QAT Deep Dives
- [qat_visual_guide.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_visual_guide.html) — 9-step beginner walkthrough of QAT using 6 concrete weights
- [qat_weight_nudge_animation.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_weight_nudge_animation.html) — Live animation: watch one weight (W=−2.30) get nudged toward INT4 bucket (−2.40) via gradient descent
- [qat_end_to_end_loop.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_end_to_end_loop.html) — Full QAT training loop (Forward → Loss → Backward/STE → Update) with graphic canvas + tutorial mode
- [w1_journey.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/w1_journey.html) — W1's complete journey step-by-step table with clickable formula breakdown
- [lr_step_estimator.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lr_step_estimator.html) — Estimate learning rate or steps needed for convergence (exponential decay formula)

---

## Progress

- ✅ Phase 1: Fine-Tuning Fundamentals (theory done) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase1_fundamentals.html)
- ✅ Phase 2: Quantization (theory done) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase2_quantization.html) | [QAT Deep Dives →](#-qat-deep-dives)
- 📖 Phase 3: LoRA & QLoRA (theory done) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase3_lora_qlora.html)
- ✅ Phase 4: Non-Instructional Fine-Tuning (hands-on done!) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase4_domain_finetune.html) | [Colab →](https://colab.research.google.com/drive/14X8AHciw9grj0BG4ojcFZMt7o87Pm9AM) | [Notebook →](Phase_4_Non_Instructional_Fine_Tuning.ipynb)
- ✅ Phase 5: Instruction Fine-Tuning (hands-on done!) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase5_instruction_finetune.html) | [Format Visual →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/instruction_format_visual.html) | [Loss Explained →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/training_loss_explained.html)
- ✅ Phase 5b: Synthetic Data Generation (theory done)
- 📖 Phase 6: Preference Alignment (DPO) (theory done) 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6_dpo_preference.html) | [DPO Loss →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/dpo_loss_simplified.html) | [Colab →](https://colab.research.google.com/drive/1fOFcaipLxzf-vaLEC1QvsFmLo-CsD1FR)
- ⬜ Phase 6B: MLOps & Deployment 📘 [Interactive →](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6b_mlops_deployment.html)
- ⬜ Phase 7: Azure AI Foundry
- ⬜ Phase 8: AWS Bedrock
- ⬜ Phase 9: AWS SageMaker
- ⬜ Phase 10: Advanced Techniques & Production
