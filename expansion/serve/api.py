"""Expansion serving API. Entirely separate process/port/namespace from the
legacy commercecore/serve/api.py `/parse-query` route, per master §1.1.

Serves the new /v1/match/* routes. Does not implement Query parsing (B01/U11
explicitly excluded per plans/12_data_and_evidence_registry.md).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CommerceCore Expansion API", version="0.1.0")

ADAPTER_PATH = Path("models/shared_adapter_v1")
_model = None
_tokenizer = None


class MatchRelevanceRequest(BaseModel):
    tenant_id: str
    query: str
    product_title: str


class MatchRelevanceResponse(BaseModel):
    request_id: str
    model_revision: str
    label: Literal["exact", "substitute", "complement", "irrelevant", "invalid_output"]
    status: Literal["ok", "invalid_input", "model_unavailable"]
    latency_ms: float


class MatchIdentityRequest(BaseModel):
    tenant_id: str
    title_a: str
    title_b: str


class MatchIdentityResponse(BaseModel):
    request_id: str
    model_revision: str
    label: Literal["same", "distinct", "unknown", "invalid_output"]
    status: Literal["ok", "invalid_input", "model_unavailable"]
    latency_ms: float


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not ADAPTER_PATH.exists():
        return None, None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16
    )
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    return _model, _tokenizer


def _generate(prompt: str, max_new_tokens: int = 8) -> str:
    import torch

    model, tokenizer = _load_model()
    if model is None:
        raise RuntimeError("adapter not available")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


@app.post("/v1/match/relevance", response_model=MatchRelevanceResponse)
def match_relevance(req: MatchRelevanceRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()
    if not req.query.strip() or not req.product_title.strip():
        raise HTTPException(status_code=400, detail="query and product_title are required")

    prompt = (
        "Classify the query-product relevance: exact, substitute, complement, or irrelevant.\n"
        f"Query: {req.query}\nProduct: {req.product_title}\nAnswer:"
    )
    try:
        raw = _generate(prompt)
    except RuntimeError:
        return MatchRelevanceResponse(
            request_id=request_id, model_revision="unavailable", label="invalid_output",
            status="model_unavailable", latency_ms=(time.time() - t0) * 1000,
        )

    label = next((l for l in ["exact", "substitute", "complement", "irrelevant"] if l in raw.lower()), "invalid_output")
    return MatchRelevanceResponse(
        request_id=request_id, model_revision="shared_adapter_v1", label=label,
        status="ok", latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/v1/match/identity", response_model=MatchIdentityResponse)
def match_identity(req: MatchIdentityRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()
    if not req.title_a.strip() or not req.title_b.strip():
        raise HTTPException(status_code=400, detail="title_a and title_b are required")

    prompt = (
        "Are these listings the same purchasable item or distinct?\n"
        f"Listing A: {req.title_a}\nListing B: {req.title_b}\nAnswer:"
    )
    try:
        raw = _generate(prompt)
    except RuntimeError:
        return MatchIdentityResponse(
            request_id=request_id, model_revision="unavailable", label="invalid_output",
            status="model_unavailable", latency_ms=(time.time() - t0) * 1000,
        )

    label = next((l for l in ["same", "distinct", "unknown"] if l in raw.lower()), "invalid_output")
    return MatchIdentityResponse(
        request_id=request_id, model_revision="shared_adapter_v1", label=label,
        status="ok", latency_ms=(time.time() - t0) * 1000,
    )


@app.get("/health")
def health():
    return {"status": "ok", "adapter_available": ADAPTER_PATH.exists()}
