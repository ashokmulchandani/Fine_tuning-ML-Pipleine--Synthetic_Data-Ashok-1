"""
DVC Stage 2: train — Fine-tune model with LoRA + SFT + MLflow tracking
=======================================================================
Called by: dvc repro train
Loads train_split.csv, trains LoRA adapter, logs everything to MLflow.

Requirements: pip install transformers peft trl datasets bitsandbytes mlflow pyyaml
"""

import json
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mlflow_tracking import TrackRun

# Load DVC params
params = yaml.safe_load(open("params.yaml"))
T = params["train"]  # Shorthand


def main():
    print("=" * 60)
    print("DVC Stage 2: Train")
    print("=" * 60)
    print(f"Model: {T['base_model']}")
    print(f"LoRA: r={T['lora_r']}, alpha={T['lora_alpha']}, dropout={T['lora_dropout']}")
    print(f"Train: {T['epochs']} epochs, batch={T['batch_size']}, lr={T['learning_rate']}")
    print()

    # ── MLflow context manager ──
    mlflow_params = {
        "base_model": T["base_model"],
        "lora_r": T["lora_r"],
        "lora_alpha": T["lora_alpha"],
        "lora_dropout": T["lora_dropout"],
        "target_modules": ",".join(T["target_modules"]),
        "quantization": T["quantization"],
        "epochs": T["epochs"],
        "batch_size": T["batch_size"],
        "gradient_accumulation": T["gradient_accumulation_steps"],
        "learning_rate": T["learning_rate"],
        "lr_scheduler": T["lr_scheduler"],
        "warmup_steps": T["warmup_steps"],
        "max_seq_length": T["max_seq_length"],
        "seed": T["seed"],
    }

    with TrackRun("instruction-ft-medical", mlflow_params) as run:
        # ── Import heavy libs inside MLflow context (lazy load) ──
        import torch
        import pandas as pd
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer

        # ── Set seed ──
        torch.manual_seed(T["seed"])

        # ── Load data ──
        train_df = pd.read_csv("data/train_split.csv")
        print(f"Loaded {len(train_df)} training examples")

        # Format as Alpaca
        def format_instruction(example):
            text = "Below is an instruction. Write a response.\n\n"
            text += "### Instruction:\n" + example["instruction"] + "\n\n"
            if example.get("input") and pd.notna(example["input"]) and str(example["input"]).strip():
                text += "### Input:\n" + example["input"] + "\n\n"
            text += "### Response:\n" + example["output"]
            return {"text": text}

        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.map(format_instruction)

        # ── Load model with quantization ──
        compute_dtype = torch.float16
        model_kwargs = {"torch_dtype": compute_dtype, "device_map": "auto"}

        if T["quantization"] in ("4bit", "8bit"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(T["quantization"] == "4bit"),
                load_in_8bit=(T["quantization"] == "8bit"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ) if T["quantization"] == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config

        tokenizer = AutoTokenizer.from_pretrained(T["base_model"])
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(T["base_model"], **model_kwargs)
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        # ── Apply LoRA ──
        lora_config = LoraConfig(
            r=T["lora_r"],
            lora_alpha=T["lora_alpha"],
            lora_dropout=T["lora_dropout"],
            target_modules=T["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        run.log_metric("trainable_params_pct",
            model.get_nb_trainable_parameters()[0] / sum(p.numel() for p in model.parameters()))

        # ── Tokenize ──
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=T["max_seq_length"],
            )

        tokenized = dataset.map(tokenize, batched=True)

        # ── Train ──
        output_dir = "./models"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=T["epochs"],
            per_device_train_batch_size=T["batch_size"],
            gradient_accumulation_steps=T["gradient_accumulation_steps"],
            learning_rate=T["learning_rate"],
            lr_scheduler_type=T["lr_scheduler"],
            warmup_steps=T["warmup_steps"],
            logging_steps=5,
            save_strategy="epoch",
            report_to="mlflow",  # ← MLflow auto-logging!
            remove_unused_columns=False,
            fp16=(T["quantization"] != "4bit"),
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized,
            formatting_func=format_instruction,
            args=training_args,
        )

        trainer.train()

        # ── Log final metrics ──
        final_loss = trainer.state.log_history[-1].get("loss", 0)
        run.log_metric("final_loss", final_loss)

        # Log loss curve
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        for step, loss_val in enumerate(losses):
            run.log_metric("loss", loss_val, step=step)

        # ── Save adapter ──
        adapter_path = Path(output_dir) / "final_adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # Copy to expected DVC output paths
        import shutil
        safetensors_src = adapter_path / "adapter_model.safetensors"
        config_src = adapter_path / "adapter_config.json"
        if safetensors_src.exists():
            shutil.copy(safetensors_src, "models/adapter_model.safetensors")
            print(f"Adapter saved: models/adapter_model.safetensors ({safetensors_src.stat().st_size} bytes)")
        if config_src.exists():
            shutil.copy(config_src, "models/adapter_config.json")

        # Register adapter in MLflow Model Registry
        run.log_adapter(str(adapter_path))

        # ── Write metrics for DVC ──
        metrics = {
            "final_loss": final_loss,
            "trainable_params": model.get_nb_trainable_parameters()[0],
            "total_steps": trainer.state.global_step,
            "best_loss": min(losses) if losses else None,
        }
        Path("metrics").mkdir(exist_ok=True)
        json.dump(metrics, open("metrics/training_metrics.json", "w"), indent=2)

        print(f"\n[PASS] Stage 2 complete. Final loss: {final_loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
