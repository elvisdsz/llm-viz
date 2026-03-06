"""
Singleton loader for GPT-2.

Usage:
    from backend.model_loader import get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer()
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from backend.config import MODEL_NAME

_model: GPT2LMHeadModel | None = None
_tokenizer: GPT2Tokenizer | None = None


def get_model_and_tokenizer() -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    global _model, _tokenizer
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {device}...")

        _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        _model.to(device)
        _model.eval()

        print("Model loaded.")

    return _model, _tokenizer
