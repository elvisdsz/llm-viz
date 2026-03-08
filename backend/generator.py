"""
Manual autoregressive generation loop.

Yields one GenerationStep per token so the WebSocket handler can stream
messages as soon as each token is produced.

use_cache=False is intentional: it forces a full forward pass every step,
giving us complete (seq_len × seq_len) attention matrices for every layer,
which are far easier to visualise than the partial KV-cache versions.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import torch

from backend.activation_extractor import extract_activations, extract_attention
from backend.config import HIDDEN_SIZE, NUM_LAYERS
from backend.model_loader import get_model_and_tokenizer


@dataclass
class GenerationStep:
    token_str: str
    token_id: int
    token_index: int          # 0-based index in the generated sequence

    # attention: weights[layer][head][key_pos], keys = ["0", "1", ...]
    attention_weights: list[list[list[float]]]
    attention_keys: list[str]

    # activations[layer] = [(neuron_idx, value), ...]
    activations: list[list[tuple[int, float]]]
    layer_count: int
    hidden_size: int


def generate_tokens(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    cancel_event: threading.Event | None = None,
):
    """
    Generator yielding GenerationStep for each produced token.

    Args:
        prompt:         text prompt to condition on
        max_new_tokens: maximum tokens to generate
        temperature:    sampling temperature (0 = greedy, >1 = more random, <1 = sharper)
        cancel_event:   optional threading.Event; generation stops early when set

    Yields:
        GenerationStep
    """
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()

    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for token_index in range(max_new_tokens):
            if cancel_event is not None and cancel_event.is_set():
                break

            outputs = model(
                generated_ids,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
            )

            # outputs.logits: (1, seq_len, vocab_size)
            logits = outputs.logits[0, -1, :]  # last position

            # Temperature sampling (temperature=0 → greedy argmax)
            if temperature == 0.0:
                next_token_id = torch.argmax(logits, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            token_str = tokenizer.decode(next_token_id)

            last_pos = generated_ids.shape[1] - 1  # before appending

            # Extract visualisation data
            attn_weights, attn_keys = extract_attention(
                outputs.attentions, last_pos
            )
            act_data = extract_activations(
                outputs.hidden_states, last_pos
            )

            step = GenerationStep(
                token_str=token_str,
                token_id=int(next_token_id),
                token_index=token_index,
                attention_weights=attn_weights,
                attention_keys=attn_keys,
                activations=act_data,
                layer_count=len(outputs.hidden_states),  # layer 0 = embedding, 1–12 = transformer blocks
                hidden_size=HIDDEN_SIZE,
            )
            yield step

            # Append token and check for EOS
            generated_ids = torch.cat(
                [generated_ids, next_token_id.unsqueeze(0)], dim=1
            )
            if eos_token_id is not None and int(next_token_id) == eos_token_id:
                break
