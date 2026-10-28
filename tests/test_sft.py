"""Tests for the supervised instruction tuning (SFT) stage."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import json
import os
import tempfile

import numpy as np
import torch

from helix.config import ModelConfig
from helix.model import HelixTransformer
from helix.tokenizer import ByteLevelBPETokenizer
from helix.sft import (
    build_instruction_pairs, tokenize_pairs, train_sft, generate_response,
    F_cross_entropy,
)


def test_build_instruction_pairs_from_dialogue(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "== DIALOGUE ==\n"
        "Alice: Can you explain recursion?\n"
        "Bob: Recursion is when a function calls itself.\n"
        "Alice: What is 2+2?\n"
        "Bob: 2+2 equals 4.\n"
    )
    pairs = build_instruction_pairs(str(corpus))
    assert len(pairs) >= 1
    instr, resp = pairs[0]
    assert "recursion" in instr.lower() or "recursion" in resp.lower()


def test_tokenize_pairs_masks_instruction():
    tok = ByteLevelBPETokenizer()
    tok.train(["INSTRUCTION: hello\nRESPONSE: world"], vocab_size=300)
    pairs = [("hello", "world")]
    out = tokenize_pairs(pairs, tok, block_size=32)
    assert len(out) == 1
    ids, mask = out[0]
    assert ids.shape == mask.shape
    # The prefix (instruction) positions should be masked; response supervised.
    assert mask.sum() > 0
    assert mask.sum() < len(mask)


def test_masked_cross_entropy_ignores_masked():
    logits = torch.zeros(1, 4, 5)
    targets = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    loss = F_cross_entropy(logits, targets, mask)
    # Uniform logits => per-token loss = ln(5); only 2 supervised positions.
    expected = (np.log(5) * 2.0) / 2.0
    assert abs(loss.item() - expected) < 1e-5


def test_sft_smoke_end_to_end(tmp_path):
    # Build a tiny base model + tokenizer and a matching config.
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world recursion fibonacci"], vocab_size=200)
    vocab_f = tmp_path / "vocab.json"
    merges_f = tmp_path / "merges.json"
    tok.save(str(vocab_f), str(merges_f))

    model = HelixTransformer(ModelConfig(
        vocab_size=len(tok), block_size=16, n_layer=1, n_head=2, n_embd=32,
        n_kv_head=1, dropout=0.0, block_type="transformer",
    ))
    base_ckpt = tmp_path / "base.pt"
    torch.save({"model": model.state_dict()}, str(base_ckpt))

    # Temp config consistent with the tiny tokenizer.
    from helix.config import HelixConfig, TrainConfig, DataConfig, save_config
    cfg = HelixConfig(
        model=ModelConfig(vocab_size=len(tok), block_size=16, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0),
        train=TrainConfig(batch_size=2, block_size=16, max_iters=1, eval_interval=1,
                          eval_iters=1, learning_rate=1e-3, warmup_iters=1,
                          lr_decay_iters=1, log_interval=1, save_interval=1,
                          out_dir=str(tmp_path / "unused")),
        data=DataConfig(train_file=str(tmp_path / "train.npy"),
                        val_file=str(tmp_path / "val.npy"),
                        vocab_file=str(vocab_f), merges_file=str(merges_f),
                        corpus_file=str(REPO / "data/seed_corpus.txt"), train_frac=0.9),
    )
    cfg_path = tmp_path / "config.json"
    save_config(cfg, str(cfg_path))

    pairs = [("What is recursion?", "Recursion is when a function calls itself."),
             ("Say hello.", "Hello!")]
    pairs_file = tmp_path / "pairs.json"
    with open(str(pairs_file), "w") as f:
        json.dump(pairs, f)

    out_dir = str(tmp_path / "sft")
    metrics = train_sft(
        config_path=str(cfg_path),
        base_checkpoint=str(base_ckpt),
        sft_pairs_file=str(pairs_file),
        out_dir=out_dir,
        max_iters=6,
        block_size=16,
        batch_size=2,
        lr=1e-3,
        device="cpu",
    )
    assert metrics["stage"] == "sft"
    assert metrics["n_pairs"] == 2
    assert metrics["actual_iters"] <= 6
    assert metrics["early_stopped"] is False
    assert metrics["patience"] == 4
    assert os.path.exists(os.path.join(out_dir, "sckpt-final.pt"))
    assert os.path.exists(os.path.join(out_dir, "sft-metrics.json"))

    # Generate a response from the SFT checkpoint.
    from helix.infer import load_model_and_tokenizer, generate
    model2, tok2, _ = load_model_and_tokenizer(str(cfg_path),
                                               os.path.join(out_dir, "sckpt-final.pt"))
    resp = generate_response(model2, tok2, "Say hello.", device="cpu", max_new_tokens=6)
    assert isinstance(resp, str) and len(resp) > 0


def test_sft_mask_supervises_exactly_response_and_eos():
    tok = ByteLevelBPETokenizer()
    tok.train(['INSTRUCTION: example\nRESPONSE: answer and answer'], vocab_size=300)
    ids, mask = tokenize_pairs([('example', 'answer')], tok, 64)[0]
    targets = ids[1:]
    supervised = targets[mask[:-1].astype(bool)].tolist()
    assert supervised == tok.encode('answer', add_eos=True)
    ids, mask = tokenize_pairs([('an instruction ' * 100, 'a response ' * 20)], tok, 16)[0]
    assert len(ids) <= 17 and mask.sum() > 0
