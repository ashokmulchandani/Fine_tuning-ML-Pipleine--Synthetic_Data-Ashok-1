"""
vLLM Serving — Production Inference for Fine-Tuned LoRA Adapters
==================================================================
Merges your LoRA adapter into the base model, then serves it with
vLLM's PagedAttention engine. 100x faster than HuggingFace pipeline().

Usage:
    python scripts/serve_vllm.py --adapter ./models/final_adapter --port 8000

Then query like OpenAI:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="./merged_model",
        messages=[{"role": "user", "content": "What is Metformin?"}]
    )

Requirements: pip install vllm transformers peft openai
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PORT = 8000
MERGED_DIR = "./merged_model"


def merge_adapter(base_model: str, adapter_path: str, output_dir: str) -> str:
    """
    Merge LoRA adapter into base model.
    Phase 3.10 Golden Rule: merge_and_unload() before deployment!
    """
    print("=" * 60)
    print("Step 1: Merge LoRA Adapter into Base Model")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  Base model: {base_model}")
    print(f"  Adapter:    {adapter_path}")

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge — bakes BxA into Wo
    print("  Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print(f"  Merged model saved: {output_dir} ({size_mb:.0f} MB)")
    print("[PASS] Adapter merged successfully.\n")
    return str(output_path)


def serve_with_vllm(model_path: str, port: int, gpu_memory: float = 0.90):
    """
    Serve the merged model with vLLM.
    Requires: pip install vllm
    """
    print("=" * 60)
    print("Step 2: Serve with vLLM")
    print("=" * 60)
    print(f"  Model:  {model_path}")
    print(f"  Port:   {port}")
    print(f"  Engine: vLLM (PagedAttention + Continuous Batching)")
    print()

    try:
        import vllm
    except ImportError:
        print("[ERROR] vLLM not installed. Run: pip install vllm")
        print("\nManual alternative:")
        print(f"  $ vllm serve {model_path} --port {port}")
        sys.exit(1)

    # Use subprocess to launch vLLM server
    import subprocess

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory),
        "--max-model-len", "2048",
        "--dtype", "float16",
    ]

    print(f"  Command: vllm serve {model_path} --port {port}")
    print(f"  API URL: http://localhost:{port}/v1")
    print(f"  Docs:    http://localhost:{port}/docs")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("[ERROR] vllm command not found. Trying alternative:")
        alt_cmd = f"vllm serve {model_path} --port {port}"
        print(f"  $ {alt_cmd}")
        print("\nYou can run this command manually in your terminal.")
        sys.exit(1)


def test_endpoint(port: int, prompt: str = "What is Metformin?"):
    """Test the vLLM server with a single query."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[INFO] Install openai to test: pip install openai")
        return

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed-for-local",
    )

    print("\n" + "=" * 60)
    print("Step 3: Test Inference")
    print("=" * 60)
    print(f"  Prompt: {prompt}")

    try:
        response = client.chat.completions.create(
            model="./merged_model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        print(f"  Response: {answer}")
        print(f"  Tokens:   {response.usage.completion_tokens}")
        print("[PASS] Inference successful!")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        print("Is the vLLM server running? Run: python scripts/serve_vllm.py --serve-only")
        print(f"Then test with: curl http://localhost:{port}/v1/models")


def main():
    parser = argparse.ArgumentParser(
        description="Serve a fine-tuned LoRA adapter with vLLM"
    )
    parser.add_argument("--adapter", type=str, default="./models/final_adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL,
                        help="Base model name or path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port for the API server")
    parser.add_argument("--gpu-memory", type=float, default=0.90,
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--serve-only", action="store_true",
                        help="Skip merging (model already merged)")
    parser.add_argument("--test", action="store_true",
                        help="Test the endpoint after starting")
    parser.add_argument("--output-dir", type=str, default=MERGED_DIR,
                        help="Output directory for merged model")

    args = parser.parse_args()

    # Step 1: Merge adapter (unless --serve-only)
    if not args.serve_only:
        model_path = merge_adapter(args.base_model, args.adapter, args.output_dir)
    else:
        model_path = args.output_dir
        if not Path(model_path).exists():
            print(f"[ERROR] Merged model not found at {model_path}")
            print("Run without --serve-only first.")
            sys.exit(1)

    # Step 2: Serve with vLLM
    serve_with_vllm(model_path, args.port, args.gpu_memory)

    # Step 3: Test (if --test)
    if args.test:
        import time
        time.sleep(5)  # Wait for server to start
        test_endpoint(args.port)


if __name__ == "__main__":
    main()
