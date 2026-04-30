"""
Simplified HuggingFace LM wrapper for TruthfulQA steering experiments.
No Hydra — uses plain Python config from config.py.
"""

from __future__ import annotations

from typing import Optional, Callable
from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import trange

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPTNeoXForCausalLM, FalconForCausalLM,
    PreTrainedTokenizer, PreTrainedModel,
    GenerationConfig,
)

from config import MODEL_NAMES, DEFAULT_CHAT_TEMPLATE, DEFAULT_GENERATION_KWARGS
from steer import get_steer_model
from steer.pace import PaCESteerer


class HuggingFaceLM:
    def __init__(
        self,
        model_name: str,
        steer_name: Optional[str] = None,
        default_generation_config: Optional[GenerationConfig] = None,
        steer_model_kwargs: dict = {},
        steer_layer_idx: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        pace_cfg: Optional[dict] = None,
    ):
        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        self.model_name = model_name
        full_name = MODEL_NAMES.get(model_name, model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            full_name, device_map=device, torch_dtype=self.dtype,
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            full_name, device_map=device, torch_dtype=self.dtype,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Match odesteer HuggingFaceLM: only install the default template when the
        # tokenizer does not provide one (pin HF/transformers for identical prompts).
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        if default_generation_config is None:
            self.default_generation_config = GenerationConfig(**DEFAULT_GENERATION_KWARGS)
        else:
            self.default_generation_config = default_generation_config

        if steer_name == "PaCE":
            if pace_cfg is None:
                raise ValueError("PaCE requires pace_cfg dict.")
            self.steer_model = PaCESteerer(pace_cfg, self.model, self.tokenizer)
        elif steer_name is not None and steer_name != "NoSteer":
            self.steer_model = get_steer_model(steer_name, **steer_model_kwargs)
        else:
            self.steer_model = None

        self.steer_layer_idx = steer_layer_idx

    def _uses_pace_hook(self) -> bool:
        return isinstance(self.steer_model, PaCESteerer)

    def generate(
        self,
        prompts: list[str],
        generation_config: Optional[GenerationConfig] = None,
        steer: bool = False,
        steer_kwargs: dict = {},
    ) -> list[str]:
        if generation_config is None:
            generation_config = self.default_generation_config

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        if steer and self.steer_model is not None:
            if self._uses_pace_hook():
                self.steer_model.register_hook()
                try:
                    outputs = self.model.generate(**inputs, generation_config=generation_config)
                finally:
                    self.steer_model.remove_hook()
            else:
                self._register_steer_hook(-1, steer_kwargs)
                outputs = self.model.generate(**inputs, generation_config=generation_config)
                self._remove_steer_hook()
        else:
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        prompt_len = inputs.attention_mask.shape[1]
        raw = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        return [r.split("\nQ:")[0] for r in raw]

    def chat(
        self,
        messages: list[list[dict]],
        generation_config: Optional[GenerationConfig] = None,
        steer: bool = False,
        steer_kwargs: dict = {},
    ) -> list[str]:
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        return self.generate(formatted, generation_config, steer, steer_kwargs)

    def fit_steer_model(self, *args, **kwargs):
        if self.steer_model is not None:
            self.steer_model.fit(*args, **kwargs)

    @torch.no_grad()
    def extract_prompt_eos_activations(
        self, prompts: list[str], layer_idx: Optional[int] = None,
    ) -> Tensor:
        if layer_idx is None:
            layer_idx = len(self.model.model.layers) // 2 - 1
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:][layer_idx]
        return hidden_states[:, -1, :]

    @torch.no_grad()
    def extract_message_eos_activations(
        self, messages: list[list[dict]], layer_idx: Optional[int] = None,
    ) -> Tensor:
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        return self.extract_prompt_eos_activations(formatted, layer_idx)

    # --- Hook machinery ---

    def _register_steer_hook(self, steer_position_idx: int, steer_kwargs: dict):
        self._hooks = []
        target = self._get_target_layer()
        handle = target.register_forward_hook(partial(
            self._steer_hook_fn,
            steer_position_idx=steer_position_idx,
            steer_kwargs=steer_kwargs,
        ))
        self._hooks.append(handle)

    def _remove_steer_hook(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _steer_hook_fn(self, module, input, output, steer_position_idx, steer_kwargs):
        hidden, reassemble = _extract_hidden(output)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        hidden = hidden.clone()
        hidden[batch_idx, steer_position_idx] = self.steer_model.steer(
            hidden[batch_idx, steer_position_idx], **steer_kwargs,
        )
        return reassemble(hidden)

    def _get_target_layer(self) -> nn.Module:
        if isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.gpt_neox.layers[self.steer_layer_idx]
        elif isinstance(self.model, FalconForCausalLM):
            return self.model.transformer.h[self.steer_layer_idx]
        else:
            return self.model.model.layers[self.steer_layer_idx]


def _extract_hidden(output) -> tuple[Tensor, Callable]:
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1:]
        def reassemble(h):
            return (h, *rest)
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
        def reassemble(h):
            output.last_hidden_state = h
            return output
    else:
        hidden = output
        def reassemble(h):
            return h
    return hidden, reassemble


def batch_chat(
    model: HuggingFaceLM,
    messages: list[list[dict]],
    T: float = 1.0,
    batch_size: int = 10,
) -> list[str]:
    num_batches = (len(messages) + batch_size - 1) // batch_size
    outputs = []
    for i in trange(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(messages))
        steer = model.steer_model is not None
        batch_out = model.chat(messages[start:end], steer=steer, steer_kwargs=dict(T=T))
        outputs.extend(batch_out)
    return outputs
