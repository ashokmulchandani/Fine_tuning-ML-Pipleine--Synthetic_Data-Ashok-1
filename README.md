# LLM Fine-Tuning & MLOps — Interactive Learning

A complete hands-on fine-tuning curriculum — from FP32 quantization to production deployment. Every phase has interactive HTML modules with code, visualizations, and Colab notebooks.

## Interactive Learning Modules

Click any link to open the interactive module. All hosted via GitHub Pages.

### 📊 Full Dashboard & Tools
| Module | Description |
|--------|-------------|
| [finetuning_plan_dashboard.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/finetuning_plan_dashboard.html) | Full dashboard — all phases, progress bar, formulas, timeline |
| [model_memory_lora_calculator.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/model_memory_lora_calculator.html) | Interactive calculator — FP32→Quant→LoRA, change any param |
| [model_deployment_guide.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/model_deployment_guide.html) | 4 levels of deployment — local → API → HuggingFace Hub |
| [training_loss_explained.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/training_loss_explained.html) | What training loss means — visual curve |

### 🎓 Phase 1: Fine-Tuning Fundamentals
| Module | Description |
|--------|-------------|
| [phase1_fundamentals.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase1_fundamentals.html) | 6 slides — 3-stage pipeline, pre-training vs FT, PEFT, FT vs RAG, data types |

### ⚙️ Phase 2: Quantization
| Module | Description |
|--------|-------------|
| [phase2_quantization.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase2_quantization.html) | 8 slides — FP32→INT4, symmetric vs asymmetric, PTQ, QAT, BitsAndBytes, BitNet |
| [bitnet_1_58bit_deep_dive.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/bitnet_1_58bit_deep_dive.html) | 9-slide deep dive into 1.58-bit LLMs (ternary weights, log₂(3), Pareto improvement) |

### 🔬 QAT Deep Dives
| Module | Description |
|--------|-------------|
| [qat_visual_guide.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_visual_guide.html) | 9-step beginner walkthrough using 6 concrete weights |
| [qat_weight_nudge_animation.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_weight_nudge_animation.html) | Live animation — watch W=−2.30 nudged toward INT4 bucket −2.40 |
| [qat_end_to_end_loop.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/qat_end_to_end_loop.html) | Full QAT training loop — Forward→Loss→Backward(STE)→Update |
| [w1_journey.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/w1_journey.html) | W1's complete journey — clickable formula breakdown per step |
| [lr_step_estimator.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lr_step_estimator.html) | Estimate lr or steps needed for convergence (exponential decay) |

### 🧬 Phase 3: LoRA & QLoRA
| Module | Description |
|--------|-------------|
| [phase3_lora_qlora.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase3_lora_qlora.html) | 10 slides — matrix decomposition, rank, equation, QLoRA, config, adapter merging |
| [lora_target_modules_explained.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_target_modules_explained.html) | Q/K/V/O visual explainer with library analogy |
| [lora_adapters_per_layer.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_adapters_per_layer.html) | Visual — 32 layers × independent adapters per module |
| [lora_alpha_dropout_explained.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_alpha_dropout_explained.html) | Alpha (volume knob) + Dropout (forget mechanism) explainer |
| [lora_adapter_merging.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/lora_adapter_merging.html) | Why merge_and_unload() between stages — stacking vs merging |

### 📄 Phase 4: Non-Instructional Fine-Tuning
| Module | Description |
|--------|-------------|
| [phase4_domain_finetune.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase4_domain_finetune.html) | 11 slides + 10-cell Colab notebook — PDF→chunk→tokenize→LoRA→Train |
| [tokenization_visual.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/tokenization_visual.html) | 4.3–4.5 visual: dataset→tokenize→labels with self-supervised learning |
| [training_memory_flow.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/training_memory_flow.html) | Where gradients & activations live during one training step |

### 💬 Phase 5: Instruction Fine-Tuning (SFT)
| Module | Description |
|--------|-------------|
| [phase5_instruction_finetune.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase5_instruction_finetune.html) | 11 slides + 6-cell Colab notebook — Alpaca format, SFTTrainer |
| [instruction_format_visual.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/instruction_format_visual.html) | Visual: how Q&A JSON becomes training strings |

### 🤖 Phase 5b: Synthetic Data Generation
| Module | Description |
|--------|-------------|
| [phase5b_synthetic_data.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase5b_synthetic_data.html) | 10 slides — Gretel AI, Mostly AI, Tonic AI, synthetic→instruction format |

### ⚖️ Phase 6: Preference Alignment (DPO)
| Module | Description |
|--------|-------------|
| [phase6_dpo_preference.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6_dpo_preference.html) | 9 slides + 5-cell Colab notebook — DPO, beta, preference pairs |
| [dpo_loss_simplified.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/dpo_loss_simplified.html) | DPO loss — teacher analogy, beta volume knob, visual demo |

### 🧪 Phase 6A: AI Testing & QA
| Module | Description |
|--------|-------------|
| [phase6a_ai_testing.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6a_ai_testing.html) | 10 slides — 4 testing layers, behavioral tests, CI/CD pipeline |

### 🔧 Phase 6B: MLOps & Deployment
| Module | Description |
|--------|-------------|
| [phase6b_mlops_deployment.html](https://ashokmulchandani.github.io/Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/phase6b_mlops_deployment.html) | 9 slides — MLflow, DVC data versioning, CI/CD, vLLM, monitoring |

---

## Phase Overview

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Fine-Tuning Fundamentals (theory) | ✅ |
| 2 | Quantization (FP32→INT4, BitsAndBytes) | ✅ |
| 3 | LoRA & QLoRA (math + implementation) | ✅ |
| 4 | Non-Instructional Fine-Tuning (domain adaptation) | ✅ |
| 5 | Instruction Fine-Tuning (SFT) | ✅ |
| 5b | Synthetic Data Generation (Gretel, Mostly AI, Tonic) | ✅ |
| 6 | Preference Alignment (DPO) | 📖 |
| 6A | AI Testing & QA (4-layer testing, CI/CD) | ⬜ |
| 6B | MLOps & Deployment (MLflow, DVC, vLLM) | ⬜ |
| 7 | Azure AI Foundry Fine-Tuning | ⬜ |
| 8 | AWS Bedrock Fine-Tuning | ⬜ |
| 9 | AWS SageMaker Fine-Tuning | ⬜ |
| 10 | Advanced Techniques | ⬜ |

## Colab Notebooks

| Phase | Colab |
|-------|-------|
| Phase 4 | [Domain Fine-Tuning](https://colab.research.google.com/drive/14X8AHciw9grj0BG4ojcFZMt7o87Pm9AM) |
| Phase 5 | Instruction Fine-Tuning (in Phase 4 notebook, later cells) |
| Phase 6 | [DPO Training](https://colab.research.google.com/drive/1fOFcaipLxzf-vaLEC1QvsFmLo-CsD1FR) |

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA, QLoRA, DoRA, IA3 |
| `trl` | SFTTrainer, DPOTrainer |
| `bitsandbytes` | 4-bit/8-bit quantization |
| `datasets` | HuggingFace dataset loading |
| `accelerate` | Multi-GPU distributed training |
| `mlflow` | Experiment tracking, model registry |
| `dvc` | Data version control |
| `vllm` | Production LLM serving |

---

**Full Plan:** [FINETUNING_PLAN.md](FINETUNING_PLAN.md)
