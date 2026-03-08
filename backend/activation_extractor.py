"""
Transform raw model outputs (hidden states + attention tensors) into
JSON-serializable lists suitable for WebSocket transmission.

All functions accept raw PyTorch tensors and return plain Python objects
so they can be serialised with Pydantic / json.dumps without extra work.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.config import TOP_N_NEURONS


def extract_attention(
    attentions: tuple[torch.Tensor, ...],
    last_pos: int,
) -> tuple[list[list[list[float]]], list[str]]:
    """
    Extract attention weights for all layers/heads at the last generated position.

    Args:
        attentions: model output attentions, shape per tensor: (1, heads, seq, seq)
        last_pos:   index of the last token in the sequence (= seq_len - 1)

    Returns:
        weights: weights[layer][head][key_position] — floats in [0, 1]
        keys:    string indices "0", "1", ... for the key positions
    """
    weights: list[list[list[float]]] = []

    for layer_attn in attentions:
        # layer_attn: (batch=1, heads, seq_len, seq_len)
        layer_attn_np = layer_attn[0].detach().cpu().float().numpy()  # (heads, seq, seq)

        head_rows: list[list[float]] = []
        for head_idx in range(layer_attn_np.shape[0]):
            # Row for the last token: shape (seq_len,)
            row = layer_attn_np[head_idx, last_pos, :].tolist()
            head_rows.append(row)

        weights.append(head_rows)

    seq_len = last_pos + 1
    keys = [str(i) for i in range(seq_len)]
    return weights, keys


def extract_activations(
    hidden_states: tuple[torch.Tensor, ...],
    last_pos: int,
    top_n: int = TOP_N_NEURONS,
) -> list[list[tuple[int, float]]]:
    """
    Extract top-N neurons (by absolute value) from each hidden state layer.

    Args:
        hidden_states: tuple of (batch=1, seq, hidden_size) tensors — one per layer
                       GPT-2 returns num_layers + 1 tensors (embedding + each block)
        last_pos:      index of the last token
        top_n:         how many neurons to keep per layer

    Returns:
        activations[layer] = list of (neuron_index, value) sorted by |value| desc
    """
    activations: list[list[tuple[int, float]]] = []

    for hs in hidden_states:
        # hs: (batch=1, seq_len, hidden_size)
        vec = hs[0, last_pos, :].detach().cpu().float().numpy()  # (hidden_size,)

        # Top-N indices by absolute value (clamped so top_n never exceeds vector length)
        abs_vec = np.abs(vec)
        n = min(top_n, len(vec))
        top_indices = np.argpartition(abs_vec, -n)[-n:]
        top_indices = top_indices[np.argsort(abs_vec[top_indices])[::-1]]

        layer_neurons: list[tuple[int, float]] = [
            (int(idx), float(vec[idx])) for idx in top_indices
        ]
        activations.append(layer_neurons)

    return activations
