"""
Pydantic models for all WebSocket messages.

The type field discriminates message direction and purpose.
All messages serialise to JSON via .model_dump_json().
"""

from __future__ import annotations

from pydantic import BaseModel


# ── Inbound (client → server) ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    type: str = "generate"
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0


# ── Outbound (server → client) ───────────────────────────────────────────────

class TokenEvent(BaseModel):
    type: str = "token"
    token_str: str
    token_id: int
    token_index: int


class AttentionData(BaseModel):
    type: str = "attention"
    token_index: int
    # keys[i] = string label for key position i (e.g. "0", "1", ...)
    keys: list[str]
    # weights[layer][head][key_position] ∈ [0, 1]
    weights: list[list[list[float]]]


class ActivationData(BaseModel):
    type: str = "activations"
    token_index: int
    layer_count: int
    hidden_size: int
    # activations[layer] = list of (neuron_idx, value) pairs
    activations: list[list[tuple[int, float]]]


class GenerationComplete(BaseModel):
    type: str = "complete"
    total_tokens: int


class ErrorEvent(BaseModel):
    type: str = "error"
    message: str
