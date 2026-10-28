"""Text generation and a tiny interactive chat interface for HELIX.

Generation runs on CPU with temperature / top-k / top-p sampling. The model is
a genuine randomly-initialized transformer; no retrieval or pretrained weights
are involved.
"""

from __future__ import annotations

import torch

from helix.config import load_config
from helix.model import HelixTransformer
from helix.tokenizer import ByteLevelBPETokenizer


def load_model_and_tokenizer(
    config_path: str,
    checkpoint_path: str,
):
    cfg = load_config(config_path)
    tok = ByteLevelBPETokenizer.load(cfg.data.vocab_file, cfg.data.merges_file)
    # Sanity: tokenizer vocab must match configured model vocab.
    if len(tok) != cfg.model.vocab_size:
        cfg.model.vocab_size = len(tok)
    model = HelixTransformer(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, tok, cfg


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
