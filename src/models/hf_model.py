"""HuggingFace model runner for MC evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from src.config import DecodingParams
from src.models.loader import load_model_and_tokenizer


@dataclass
class GenerationResult:
    output_text: str
    token_count: int
    latency_ms: float


class HFModelRunner:
    """Wraps a HuggingFace causal LM for deterministic text generation."""

    def __init__(
        self,
        model_id: str,
        decoding: DecodingParams | None = None,
        quantize_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.decoding = decoding or DecodingParams()
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_id, quantize_4bit=quantize_4bit
        )
        self.device = next(self.model.parameters()).device

    def _apply_chat_template(self, prompt: str) -> str:
        """Wrap prompt in the model's chat template if available."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompt: str) -> GenerationResult:
        """Generate text from a single prompt deterministically."""
        torch.manual_seed(self.decoding.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.decoding.seed)
        elif self.device.type == "mps":
            torch.mps.manual_seed(self.decoding.seed)

        formatted_prompt = self._apply_chat_template(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.decoding.max_new_tokens,
                temperature=self.decoding.temperature,
                do_sample=self.decoding.do_sample,
                top_p=self.decoding.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        new_tokens = output_ids[0][input_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = len(new_tokens)

        return GenerationResult(
            output_text=output_text,
            token_count=token_count,
            latency_ms=latency_ms,
        )

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate text for a batch of prompts.

        Falls back to sequential generation for simplicity and to avoid
        padding-related artifacts in deterministic decoding.
        """
        return [self.generate(p) for p in prompts]
