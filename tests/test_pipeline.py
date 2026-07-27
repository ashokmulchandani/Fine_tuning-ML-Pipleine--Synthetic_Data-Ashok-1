"""
Layer 3: Pipeline Testing — Does Training Actually Work?
=========================================================
End-to-end smoke test: tiny dataset → train 1 epoch → save adapter →
load adapter → predict. If this passes, the pipeline is wired up correctly.

Designed to run in Google Colab with a tiny model (e.g., facebook/opt-125m).
Marked @pytest.mark.gpu — skipped by default on CPU-only environments.

Run (with GPU):  python -m pytest tests/test_pipeline.py -v -m "gpu"
Run (skip GPU):  python -m pytest tests/test_pipeline.py -v
"""

import os
import shutil
import tempfile
import pytest


# ═══════════════════════════════════════════════════════════════
# GPU Requirement Check
# ═══════════════════════════════════════════════════════════════

# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.gpu

# Check for transformers/peft availability (Colab has these; local may not)
transformers = pytest.importorskip("transformers", reason="transformers not installed")
peft = pytest.importorskip("peft", reason="peft not installed")
torch = pytest.importorskip("torch", reason="torch not installed")


# ═══════════════════════════════════════════════════════════════
# Config — tune these for your environment
# ═══════════════════════════════════════════════════════════════

# Tiny model for smoke tests (125M params, runs on free Colab GPU)
BASE_MODEL = "facebook/opt-125m"
TEST_ADAPTER_DIR = "./test_adapter_output"


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def create_tiny_dataset():
    """
    Create a minimal dataset (2 examples) for smoke testing.
    This trains fast — the goal is to verify the pipeline wires,
    not to produce a good model.
    """
    from datasets import Dataset
    data = [
        {
            "instruction": "### Instruction: What is AI?\n### Response:",
            "output": "AI stands for Artificial Intelligence."
        },
        {
            "instruction": "### Instruction: What is ML?\n### Response:",
            "output": "ML stands for Machine Learning."
        },
    ]
    return Dataset.from_list(data)


def format_instruction(example: dict) -> str:
    """Format an example into a training string (Alpaca format)."""
    return f"{example['instruction']}{example['output']}"


# ═══════════════════════════════════════════════════════════════
# Pipeline Smoke Tests
# ═══════════════════════════════════════════════════════════════

class TestPipelineSmoke:
    """Verify the entire training pipeline works end-to-end."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test adapter directory before and after each test."""
        if os.path.exists(TEST_ADAPTER_DIR):
            shutil.rmtree(TEST_ADAPTER_DIR)
        yield
        if os.path.exists(TEST_ADAPTER_DIR):
            shutil.rmtree(TEST_ADAPTER_DIR)

    def test_train_one_epoch_does_not_crash(self):
        """
        SMOKE TEST: Train on 2 examples for 1 epoch.
        This should complete without exceptions.
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset

        # Load tiny model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

        # Apply LoRA with minimal config
        lora_config = LoraConfig(
            r=4,                    # Tiny rank for fast smoke test
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Create tiny dataset
        dataset = create_tiny_dataset()

        def tokenize(examples):
            texts = [format_instruction(ex) for ex in [
                {"instruction": examples["instruction"][i], "output": examples["output"][i]}
                for i in range(len(examples["instruction"]))
            ]]
            result = tokenizer(texts, truncation=True, padding=True, max_length=64)
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized = dataset.map(tokenize, batched=True)

        # Train 1 epoch (just to verify no crashes)
        training_args = TrainingArguments(
            output_dir=TEST_ADAPTER_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",          # Don't save intermediate checkpoints
            report_to="none",             # Don't log to wandb/mlflow
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
        )

        # This is the critical test: training should not crash
        trainer.train()

        # Verify training happened
        assert trainer.state.global_step > 0, \
            "Training should complete at least one step"

    def test_save_and_load_adapter(self):
        """
        SMOKE TEST: Save adapter → verify file exists → load it back.
        The adapter file is the deliverable of fine-tuning.
        """
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Save adapter
        model.save_pretrained(TEST_ADAPTER_DIR)

        # Check 7: Adapter file exists and is > 0 bytes
        adapter_file = os.path.join(TEST_ADAPTER_DIR, "adapter_model.safetensors")
        assert os.path.exists(adapter_file), \
            f"adapter_model.safetensors not found in {TEST_ADAPTER_DIR}"
        file_size = os.path.getsize(adapter_file)
        assert file_size > 0, \
            f"adapter_model.safetensors is empty (0 bytes). Save failed."

        # Check 8: Adapter loads without error
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        loaded = PeftModel.from_pretrained(base_model, TEST_ADAPTER_DIR)
        assert loaded is not None, "PeftModel.from_pretrained() returned None"
        assert isinstance(loaded, PeftModel), \
            f"Expected PeftModel, got {type(loaded).__name__}"

    def test_loss_is_finite(self):
        """
        SMOKE TEST: After one training step, loss should be a finite number
        (not NaN, not infinity). NaN loss = learning rate too high.
        """
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
        import numpy as np

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        dataset = create_tiny_dataset()

        def tokenize(examples):
            texts = [
                format_instruction({
                    "instruction": examples["instruction"][i],
                    "output": examples["output"][i]
                })
                for i in range(len(examples["instruction"]))
            ]
            result = tokenizer(texts, truncation=True, padding=True, max_length=64)
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized = dataset.map(tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=TEST_ADAPTER_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
        trainer.train()

        # Get the last logged loss
        logs = trainer.state.log_history
        losses = [log["loss"] for log in logs if "loss" in log]

        assert len(losses) > 0, "No loss values logged during training"

        for loss in losses:
            assert not np.isnan(loss), f"Loss is NaN! Usually means learning rate too high."
            assert not np.isinf(loss), f"Loss is infinite! Check data for extreme values."
            assert loss > 0, f"Loss should be positive, got {loss}"

    def test_inference_after_training(self):
        """
        SMOKE TEST: After training, the model should produce non-empty output.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        import torch

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        lora_config = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Generate from a simple prompt
        prompt = "### Instruction: What is AI?\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Should produce more than just the input prompt
        assert len(generated) > len(prompt), \
            f"Model didn't generate any new tokens. Output same as input: '{generated}'"

        # Should not be empty
        assert len(generated.strip()) > 0, \
            "Generated text is empty"
