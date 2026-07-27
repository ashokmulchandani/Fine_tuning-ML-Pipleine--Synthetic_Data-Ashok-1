"""
Phase 6A: Shared Pytest Fixtures for Fine-Tuning Tests
======================================================
These fixtures simulate fine-tuning data and model outputs so tests
run fast without a real GPU or model. Each fixture returns data in
the Alpaca instruction format used throughout this project.

FORMAT (from CLAUDE.md):
    ### Instruction:\n...\n### Input:\n...\n### Response:\n...
"""

import pytest
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Layer 1 Fixtures: Training/Validation Data
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_instruction_data():
    """
    3 clean instruction examples for the medical Q&A domain.
    Used in: test_data.py (happy path), test_pipeline.py (smoke test).
    """
    return pd.DataFrame([
        {
            "instruction": "### Instruction: What is Metformin?\n### Response:",
            "input": "",
            "output": "Metformin is a biguanide medication used to treat type 2 diabetes. "
                      "It works by decreasing glucose production in the liver and improving "
                      "insulin sensitivity."
        },
        {
            "instruction": "### Instruction: List side effects of Metformin.\n### Response:",
            "input": "",
            "output": "Common side effects include nausea, diarrhea, stomach upset, "
                      "and a metallic taste. Rare but serious: lactic acidosis."
        },
        {
            "instruction": "### Instruction: What is the maximum daily dose of Metformin?\n### Response:",
            "input": "",
            "output": "The maximum recommended daily dose is 2,550 mg for adults, "
                      "typically divided into 2-3 doses with meals."
        },
    ])


@pytest.fixture
def bad_instruction_data():
    """
    Data with deliberate problems: nulls, empty strings, malformed instruction.
    Used in: test_data.py (edge cases and failure modes).
    """
    return pd.DataFrame([
        {
            "instruction": "### Instruction: What is Metformin?\n### Response:",
            "input": "",
            "output": "Metformin is a biguanide."  # ✅ Valid row
        },
        {
            "instruction": None,                     # ❌ Null instruction
            "input": "",
            "output": "Some answer"
        },
        {
            "instruction": "### Instruction: Side effects?\n### Response:",
            "input": "",
            "output": ""                             # ❌ Empty output
        },
        {
            "instruction": "",                       # ❌ Empty instruction
            "input": "",
            "output": "Some answer"
        },
    ])


# ═══════════════════════════════════════════════════════════════
# Layer 1 Fixtures: Preference Data (DPO format)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_preference_data():
    """
    Preference pairs for DPO training.
    Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    return pd.DataFrame([
        {
            "prompt": "### Instruction: What is Metformin?\n### Response:",
            "chosen": "Metformin is a biguanide medication that lowers blood glucose "
                      "by reducing hepatic glucose production and increasing insulin sensitivity.",
            "rejected": "Metformin is a pill. Take it with water."
        },
        {
            "prompt": "### Instruction: Side effects of Metformin?\n### Response:",
            "chosen": "Common side effects include gastrointestinal issues like nausea, "
                      "diarrhea, and abdominal discomfort. Rarely, lactic acidosis may occur.",
            "rejected": "It might make your stomach hurt or something, idk."
        },
    ])


# ═══════════════════════════════════════════════════════════════
# Layer 2 Fixtures: Mock Model Responses
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_pipeline_response():
    """
    Simulates the output of transformers.pipeline() for a medical Q&A model.
    Returns a dict with the same structure as pipeline(...)[0].
    """
    return {
        "generated_text": "### Instruction: What is Metformin?\n### Response:\n"
                          "Metformin is a biguanide medication used to treat type 2 "
                          "diabetes. It helps control blood sugar levels."
    }


@pytest.fixture
def mock_dpo_pipeline():
    """
    Returns a callable that mimics a DPO-trained model pipeline.
    The callable accepts a prompt string and returns a list of dicts
    matching the transformers pipeline output format.
    """
    def _mock_pipe(prompt: str) -> list:
        prompt_lower = prompt.lower()

        # DIR: refuse harmful prompts
        if any(word in prompt_lower for word in ["overdose", "harm", "illegal"]):
            return [{"generated_text": "I cannot provide information about harmful actions."}]

        # MFT: answer Metformin questions
        if "metformin" in prompt_lower:
            if "side effect" in prompt_lower:
                return [{"generated_text": "Common side effects of Metformin include nausea, "
                                           "diarrhea, gastrointestinal discomfort, and a metallic taste. "
                                           "Rare but serious: lactic acidosis."}]
            return [{"generated_text": "Metformin is a biguanide medication for type 2 diabetes. "
                                       "It reduces glucose production in the liver."}]

        # MFT: answer biguanide drug questions
        if "biguanide" in prompt_lower:
            return [{"generated_text": "The main biguanide drug is Metformin. It is used "
                                       "to treat type 2 diabetes by reducing hepatic glucose production."}]

        # MFT: general medical questions
        if any(word in prompt_lower for word in ["diabetes", "insulin", "a1c"]):
            return [{"generated_text": "This is a medical question about endocrinology. "
                                       "Diabetes is a condition affecting blood sugar regulation."}]

        # Fallback
        return [{"generated_text": "I don't have enough information to answer that question."}]

    return _mock_pipe
