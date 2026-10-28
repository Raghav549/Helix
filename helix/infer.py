"""Text generation and a tiny interactive chat interface for HELIX.

Generation runs on CPU with temperature / top-k / top-p sampling. Three modes:
  * full-context recompute (`generate`) — the baseline, simple and correct
  * KV-cached incremental generation (`generate_with_cache`) — avoids
    recomputing the prompt each step; a CPU-performance hypothesis to benchmark
  * quantized generation (`generate_quantized`) — int8 weights, measured against
    float32; speedup not assumed

The model is a genuinely randomly-initialized transformer; no retrieval or
pretrained weights are involved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from helix.config import load_config
from helix.kvcache import KVCache
from helix.model import HelixTransformer
from helix.quantize import quantize_model
from helix.tokenizer import ByteLevelBPETokenizer


def load_model_and_tokenizer(
    config_path: str,
    checkpoint_path: str,
):
    cfg = load_config(config_path)
    tok = ByteLevelBPETokenizer.load(cfg.data.vocab_file, cfg.data.merges_file)
    if len(tok) != cfg.model.vocab_size:
        cfg.model.vocab_size = len(tok)
    model = HelixTransformer(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, tok, cfg


def load_quantized_model_and_tokenizer(config_path: str, checkpoint_path: str):
    """Load a checkpoint, then quantize Linear weights to int8 (CPU inference)."""
    model, tok, cfg = load_model_and_tokenizer(config_path, checkpoint_path)
    model = quantize_model(model)
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def _sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample the next token id from logits (B, vocab). Returns (B, 1)."""
    logits = logits[:, -1, :] / max(temperature, 1e-6)
    vocab = logits.size(-1)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab)
        v, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < v[:, [-1]], float("-inf"), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: HelixTransformer,
    tok: ByteLevelBPETokenizer,
    text: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """Full-context-recompute generation (baseline, correct on all hardware)."""
    ids = tok.encode(text)
    if not ids:
        ids = [tok.vocab.get(0, 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return tok.decode(out[0].tolist())


@torch.no_grad()
def generate_with_cache(
    model: HelixTransformer,
    tok: ByteLevelBPETokenizer,
    text: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """KV-cached incremental generation.

    Primes the cache with the full prompt, then generates one token per step
    using only the new token plus the cached K/V. Should produce identical
    output to `generate` (same sampling) while recomputing less.
    """
    model.eval()
    ids = tok.encode(text)
    if not ids:
        ids = [tok.vocab.get(0, 0)]
    # Truncate prompt to fit the cache / context window.
    if len(ids) > model.config.block_size:
        ids = ids[-model.config.block_size :]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    head_dim = model.config.n_embd // model.config.n_head
    cache = KVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        max_seq=model.config.block_size,
        n_kv_head=model.config.n_kv_head,
        head_dim=head_dim,
        device=device,
    )

    # Prime: process the whole prompt to fill the cache.
    logits, cache = model.incremental_forward(idx, cache, 0)
    next_id = _sample_next(logits, temperature, top_k, top_p)
    generated = [next_id.item()]

    pos = len(ids)
    for _ in range(max_new_tokens - 1):
        logits, cache = model.incremental_forward(next_id, cache, pos)
        next_id = _sample_next(logits, temperature, top_k, top_p)
        generated.append(next_id.item())
        pos += 1

    all_ids = ids + generated
    return tok.decode(all_ids)


@torch.no_grad()
def generate_quantized(
    model: HelixTransformer,
    tok: ByteLevelBPETokenizer,
    text: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
    use_cache: bool = True,
) -> str:
    """Generation on an int8-quantized model (full recompute or cached)."""
    if use_cache:
        return generate_with_cache(model, tok, text, max_new_tokens,
                                   temperature, top_k, top_p, device)
    return generate(model, tok, text, max_new_tokens,
                    temperature, top_k, top_p, device)


def chat(config_path: str, checkpoint_path: str) -> None:
    model, tok, cfg = load_model_and_tokenizer(config_path, checkpoint_path)
    device = cfg.train.device
    print("HELIX chat (CPU baseline). Type your message; Ctrl-C or 'exit' to quit.\n")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in ("exit", "quit"):
                break
            reply = generate(
                model,
                tok,
                user,
                max_new_tokens=60,
                temperature=0.8,
                top_k=50,
                device=device,
            )
            print(f"helix> {reply}\n")
    except KeyboardInterrupt:
        pass
    print("Goodbye.")
