# LLM Fine-Tuning & Synthetic Data Pipeline

A complete hands-on fine-tuning learning project covering theory, quantization, LoRA/QLoRA, instruction tuning, preference alignment (DPO), synthetic data generation (Gretel, Mostly AI, Tonic), and cloud fine-tuning across Azure AI Foundry, AWS Bedrock, and AWS SageMaker.

## Repo Structure

```
Fine_tuning-ML-Pipleine--Synthetic_Data-Ashok-1/
├── FINETUNING_PLAN.md                  # Master plan (all phases)
├── FINE_Tuning_Transcripts/            # Reference transcripts
├── Phase_1_Fundamentals/               # Fine-tuning theory
├── Phase_2_Quantization/               # FP32→INT4, BitsAndBytes
├── Phase_3_LoRA_QLoRA/                 # LoRA math + QLoRA implementation
├── Phase_4_NonInstructional/           # Domain adaptation from PDFs
├── Phase_5_InstructionFT/              # SFT with instruction datasets
├── Phase_5b_Synthetic_Data/            # Gretel AI, Mostly AI, Tonic
├── Phase_6_DPO/                        # Preference alignment (DPO/RLHF)
├── Phase_7_Azure_Foundry/              # Azure AI Foundry fine-tuning
├── Phase_8_Bedrock/                    # AWS Bedrock fine-tuning
├── Phase_9_SageMaker/                  # AWS SageMaker fine-tuning
├── Phase_10_Advanced/                  # DoRA, IA3, Prefix Tuning, vLLM
├── notebooks/                          # Jupyter notebooks
├── data/                               # Training datasets (gitignored)
└── models/                             # Saved adapters/weights (gitignored)
```

## Phase Overview

| Phase | Topic | Platform |
|-------|-------|----------|
| 1 | Fine-Tuning Fundamentals (theory) | Theory |
| 2 | Quantization (FP32→FP16→INT8→INT4, BitsAndBytes) | Theory + Colab |
| 3 | LoRA & QLoRA (math + implementation) | Colab |
| 4 | Non-Instructional Fine-Tuning (domain adaptation from PDFs) | Colab |
| 5 | Instruction Fine-Tuning (SFT, Alpaca/ChatML format) | Colab |
| 5b | Synthetic Data Generation (Gretel, Mostly AI, Tonic) | Colab + Cloud |
| 6 | Preference Alignment (DPO/RLHF) | Colab |
| 7 | Azure AI Foundry Fine-Tuning (GPT-4o, no-code) | Azure |
| 8 | AWS Bedrock Fine-Tuning (Nova Micro) | AWS |
| 9 | AWS SageMaker Fine-Tuning (QLoRA + Spectrum, MLflow) | AWS |
| 10 | Advanced PEFT (DoRA, IA3, Prefix Tuning, vLLM deployment) | Colab |

## Synthetic Data Tools (Phase 5b)

| Tool | Best For | Privacy |
|------|----------|---------|
| **Gretel AI** | Tabular + text + time-series synthetic data | GDPR/HIPAA safe |
| **Mostly AI** | Financial/CRM tabular data | Mathematically provable privacy |
| **Tonic AI** | Dev/test DB cloning (Postgres/Supabase) | Referential integrity preserved |

## Key Libraries

| Library | Purpose |
|---------|---------|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA, QLoRA, DoRA, IA3 |
| `trl` | SFTTrainer, DPOTrainer, PPOTrainer |
| `bitsandbytes` | 4-bit/8-bit quantization |
| `datasets` | HuggingFace dataset loading |
| `accelerate` | Multi-GPU distributed training |
| `mlflow` | Experiment tracking |
| `gretel-client` | Synthetic data generation |
| `vllm` / `tgi` | Production LLM serving |

## Progress

- ⬜ Phase 1: Fine-Tuning Fundamentals
- ⬜ Phase 2: Quantization
- ⬜ Phase 3: LoRA & QLoRA
- ⬜ Phase 4: Non-Instructional Fine-Tuning
- ⬜ Phase 5: Instruction Fine-Tuning
- ⬜ Phase 5b: Synthetic Data Generation
- ⬜ Phase 6: Preference Alignment (DPO)
- ⬜ Phase 7: Azure AI Foundry
- ⬜ Phase 8: AWS Bedrock
- ⬜ Phase 9: AWS SageMaker
- ⬜ Phase 10: Advanced Techniques

## Transcript Coverage

| Transcript | Covers |
|------------|--------|
| Krish Naik - Fine Tuning LLM Models | Phase 2 (quantization), Phase 3 (LoRA/QLoRA), Phase 4 (Llama 2) |
| LLM Fine-Tuning Crash Course (Sunny) | Phase 1, Phase 4, Phase 5, Phase 6 (DPO) |
| Azure AI Foundry - Step by Step | Phase 7 (GPT-3.5 Turbo) |
| Azure AI Foundry - Kirk | Phase 7 (GPT-4o, cost breakdown) |
| Amazon Bedrock Fine-Tuning | Phase 8 (Nova Micro, S3, hyperparameters) |
| AWS SageMaker Fine-Tuning | Phase 9 (QLoRA vs Spectrum, MLflow) |

## Related Repos

| Repo | Purpose |
|------|---------|
| [CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1](https://github.com/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1) | CUDA fundamentals, computer vision, 3D perception |
| [AWS-Agentcore-bedrock-Ashok-1](https://github.com/ashokmulchandani/AWS-Agentcore-bedrock-Ashok-1) | AWS Bedrock agents, AgentCore |

## License

Private — internal use.
