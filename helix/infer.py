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

import codecs
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
    from helix.runtime import configure_threads
    configure_threads()
    cfg = load_config(config_path)
    tok = ByteLevelBPETokenizer.load(cfg.data.vocab_file, cfg.data.merges_file)
    if len(tok) != cfg.model.vocab_size:
        cfg.model.vocab_size = len(tok)
    model = HelixTransformer(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(cfg.train.device)
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
    logits = logits[:, -1, :]
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
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
        ids = [tok._id_for_bytes(b"<|endoftext|>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    decoded = tok.decode(out[0].tolist())
    if not text and decoded.startswith("<|endoftext|>"):
        decoded = decoded[len("<|endoftext|>"):]
    return decoded


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
        ids = [tok._id_for_bytes(b"<|endoftext|>")]
    if max_new_tokens <= 0:
        return text
    # Other experimental block types do not have a mathematically equivalent
    # transformer KV-cache. Fall back to correct full recomputation.
    if model.config.block_type != "transformer":
        return generate(model, tok, text, max_new_tokens, temperature, top_k, top_p, device)
    all_ids = list(ids)
    head_dim = model.config.n_embd // model.config.n_head
    cache = KVCache(n_layers=model.config.n_layer, batch_size=1,
                    max_seq=model.config.block_size, n_kv_head=model.config.n_kv_head,
                    head_dim=head_dim, device=device, dtype=model.tok_emb.weight.dtype)
    context = all_ids[-model.config.block_size:]
    idx = torch.tensor([context], dtype=torch.long, device=device)
    logits, cache = model.incremental_forward(idx, cache, 0)
    for step in range(max_new_tokens):
        next_id = _sample_next(logits, temperature, top_k, top_p)
        all_ids.append(next_id.item())
        if step == max_new_tokens - 1:
            break
        if cache.seq_len(0) >= model.config.block_size:
            # Recompute a sliding window, matching full-context generation. A
            # simple eviction would leave stale attention-dependent KV values.
            cache.reset()
            idx = torch.tensor([all_ids[-model.config.block_size:]], dtype=torch.long, device=device)
            logits, cache = model.incremental_forward(idx, cache, 0)
        else:
            logits, cache = model.incremental_forward(next_id, cache, cache.seq_len(0))
    decoded = tok.decode(all_ids)
    if not text and decoded.startswith("<|endoftext|>"):
        decoded = decoded[len("<|endoftext|>"):]
    return decoded


def stream_generate(model: HelixTransformer, tok: ByteLevelBPETokenizer, text: str,
                    max_new_tokens: int = 80, temperature: float = .8,
                    top_k: int | None = 40, top_p: float | None = None,
                    device: str = "cpu"):
    """Yield actual decoded text as tokens are sampled; stop on reserved EOS.

    Uses incremental KV cache for transformer blocks and an exact sliding
    window re-prime when full. A byte incremental decoder preserves UTF-8 across
    token boundaries. Experimental non-transformer blocks use recomputation.
    """
    if max_new_tokens < 0 or max_new_tokens > 2048:
        raise ValueError("max_new_tokens must be between 0 and 2048")
    eos = tok._id_for_bytes(b"<|endoftext|>")
    reserved = {tok._id_for_bytes(t.encode()) for t in tok.special_tokens}
    ids = tok.encode(text) or [eos]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    model.eval()
    use_cache = model.config.block_type == "transformer"
    with torch.no_grad():
        cache = None
        if use_cache:
            cache = KVCache(model.config.n_layer, 1, model.config.block_size,
                            model.config.n_kv_head, model.config.n_embd // model.config.n_head,
                            device, dtype=model.tok_emb.weight.dtype)
            idx = torch.tensor([ids[-model.config.block_size:]], dtype=torch.long, device=device)
            logits, _ = model.incremental_forward(idx, cache, 0)
        for step in range(max_new_tokens):
            if not use_cache:
                idx = torch.tensor([ids[-model.config.block_size:]], dtype=torch.long, device=device)
                logits, _ = model(idx)
            next_id = _sample_next(logits, temperature, top_k, top_p)
            token = int(next_id.item())
            if token == eos:
                break
            ids.append(token)
            if token not in reserved:
                chunk = decoder.decode(tok.vocab[token], final=False)
                if chunk:
                    yield chunk
            if step == max_new_tokens - 1:
                break
            if use_cache:
                if cache.seq_len(0) >= model.config.block_size:
                    cache.reset()
                    idx = torch.tensor([ids[-model.config.block_size:]], dtype=torch.long, device=device)
                    logits, _ = model.incremental_forward(idx, cache, 0)
                else:
                    logits, _ = model.incremental_forward(next_id, cache, cache.seq_len(0))
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


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
            prompt = f"INSTRUCTION: {user}\nRESPONSE: "
            print("helix> ", end="", flush=True)
            for chunk in stream_generate(model, tok, prompt, max_new_tokens=80,
                                         temperature=0.7, top_k=40, device=device):
                print(chunk, end="", flush=True)
            print("\n")
    except KeyboardInterrupt:
        pass
    print("Goodbye.")
