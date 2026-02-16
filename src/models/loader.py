"""Model and tokenizer loading utilities."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_id: str,
    quantize_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load a HuggingFace causal LM and its tokenizer.

    Automatically selects CUDA, MPS, or CPU.  4-bit quantization is only
    supported on CUDA and will be skipped with a warning on other devices.

    Args:
        model_id: HuggingFace model identifier.
        quantize_4bit: If True, load in NF4 4-bit quantization (CUDA only).

    Returns:
        (model, tokenizer) tuple.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "dtype": "auto",
    }

    if quantize_4bit and device.type == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    elif quantize_4bit:
        print(f"  Warning: 4-bit quantization not supported on {device.type}, loading in fp16.")

    if device.type == "cuda" and "device_map" not in kwargs:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    # For MPS/CPU, explicitly move the model to the device
    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    print(f"  Model loaded on: {device}")

    return model, tokenizer
