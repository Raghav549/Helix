"""Supervised instruction tuning (SFT) — a stage strictly separate from pretraining.

Pretraining (helix/train.py) teaches next-token prediction on a broad corpus.
SFT (this module) teaches instruction following: given an instruction, produce
the desired response. It loads a base checkpoint trained from random init and
continues training on (instruction, response) pairs, supervising ONLY the
response portion of each sequence (instruction tokens are masked out of the
loss). Model weights, optimizer state, and data are versioned separately from
the pretraining run so the two stages can be tested and improved independently.

CPU-first; tiny models so a full SFT run completes on an ordinary CPU.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import torch

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, load_config, save_config
from helix.data import get_batch, load_ids
from helix.infer import load_model_and_tokenizer
from helix.model import HelixTransformer
from helix.tokenizer import ByteLevelBPETokenizer


# ---------------------------------------------------------------------------
# Instruction-pair extraction from a corpus
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_instruction_pairs(corpus_file: str) -> List[Tuple[str, str]]:
    """Extract (instruction, response) pairs from a structured corpus.

    Heuristics (deterministic, no external deps):
      * Dialogue: "Speaker: question" followed by "Listener: answer" -> QA pair.
      * Explicit Q&A: lines starting with "Q:" / "Question:" answered by the
        next non-speaker line.
      * Declarative sentences -> synthetic "What is ..." / explanation pairs
        when a sentence contains a definitional verb (is/are/are/was/means).

    Only pairs where both sides are non-trivial are kept. Returns a deduplicated
    list. Intended for small seed corpora; the general ingestion pipeline
    (helix/corpus.py) supplies larger sources in later stages.
    """
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    pairs: List[Tuple[str, str]] = []
    seen = set()

    # Split into logical lines; drop section headers like "== SCIENCE ==".
    raw_lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("==")]

    dialogue_re = re.compile(r"^([A-Za-z][A-Za-z0-9 ]+?):\s*(.+)$")
    q_re = re.compile(r"^(?:Q|Question)[:\.\-]\s*(.+)$", re.IGNORECASE)

    pending_question: str | None = None
    for line in raw_lines:
        m = dialogue_re.match(line)
        if m:
            speaker, utt = m.group(1).strip(), _clean(m.group(2))
            if pending_question is not None and utt:
                key = (pending_question, utt)
                if key not in seen:
                    seen.add(key)
                    pairs.append(("Answer: " + pending_question, utt))
                pending_question = None
            else:
                # Treat a question-like utterance as a pending instruction.
                low = utt.lower()
                if low.endswith("?") or low.startswith(("how ", "why ", "what ", "can you ",
                                                        "could you ", "explain ")):
                    pending_question = utt
                else:
                    pending_question = None
            continue

        mq = q_re.match(line)
        if mq:
            pending_question = _clean(mq.group(1))
            continue

        # Declarative continuation: if a question is pending, this line answers it.
        if pending_question is not None and line:
            utt = _clean(line)
            if len(utt) > 3 and not utt.startswith(("Alice:", "Bob:")):
                key = (pending_question, utt)
                if key not in seen:
                    seen.add(key)
                    pairs.append(("Answer: " + pending_question, utt))
                pending_question = None

    # Fallback synthetic pairs from definitional sentences if the corpus has
    # no dialogue/Q&A structure.
    if not pairs:
        for sent in re.split(r"[.!?]+", text):
            sent = _clean(sent)
            if len(sent) < 8:
                continue
            low = sent.lower()
            if any(w in low for w in (" is ", " are ", " means ", " was ", " were ")):
                topic = sent.split(" is ")[0].strip()[:40]
                instr = f"What is {topic}?" if topic else f"Explain: {sent[:40]}"
                key = (instr, sent)
                if key not in seen:
                    seen.add(key)
                    pairs.append((instr, sent))

    return pairs


# ---------------------------------------------------------------------------
# SFT dataset: tokenize pairs, mask instruction portion of the loss
# ---------------------------------------------------------------------------

def tokenize_pairs(pairs: List[Tuple[str, str]], tok: ByteLevelBPETokenizer,
                   block_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert pairs into (input_ids, loss_mask) arrays.

    loss_mask[i] == 1 means the token at position i (predicting token i+1) is
    supervised. Instruction tokens and padding are masked (0); only response
    tokens contribute to the loss.
    """
    eos = "<|endoftext|>"
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for instr, resp in pairs:
        text = f"INSTRUCTION: {instr}\nRESPONSE: {resp} {eos}"
        ids = tok.encode(text, add_eos=False)
        # Find where the response portion begins.
        prefix = f"INSTRUCTION: {instr}\nRESPONSE: "
        prefix_ids = tok.encode(prefix, add_eos=False)
        n_prefix = len(prefix_ids)
        if len(ids) < 2 or n_prefix >= len(ids):
            continue
        ids = ids[:block_size]
        mask = np.zeros(len(ids), dtype=np.float32)
        # Supervise tokens that predict response tokens: positions >= n_prefix.
        # Position i predicts token i+1; mask position i if token i+1 is in the
        # response region.
        for i in range(n_prefix - 1, len(ids) - 1):
            mask[i] = 1.0
        # Always supervise the final token if it predicts EOS inside response.
        if len(ids) - 1 >= n_prefix - 1:
            mask[len(ids) - 1] = 1.0
        out.append((np.array(ids, dtype=np.int64), mask))
    return out


# ---------------------------------------------------------------------------
# SFT training loop (separate stage from pretraining)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sft_eval_loss(model: HelixTransformer, batches, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    for ids, mask in batches:
        x = torch.tensor(np.asarray(ids[:-1])[None, :], dtype=torch.long, device=device)
        y = torch.tensor(np.asarray(ids[1:])[None, :], dtype=torch.long, device=device)
        m = torch.tensor(np.asarray(mask[:-1])[None, :], dtype=torch.float32, device=device)
        logits, _ = model(x)
        loss = F_cross_entropy(logits, y, m)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(count, 1)


def F_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy: only positions with mask==1 contribute."""
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
    )
    loss = loss * mask.view(-1)
    denom = mask.view(-1).sum().clamp_min(1.0)
    return loss.sum() / denom


def train_sft(config_path: str, base_checkpoint: str, sft_pairs_file: str,
              out_dir: str, max_iters: int = 100, eval_interval: int = 25,
              batch_size: int = 4, block_size: int = 128, lr: float = 1e-4,
              seed: int = 1337, device: str = "cpu") -> dict:
    """Run SFT from a base pretrained-from-scratch checkpoint.

    Returns a metrics dict and writes checkpoints + log to out_dir.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path)
    cfg.train.device = device
    cfg.train.block_size = block_size
    cfg.train.batch_size = batch_size
    cfg.train.learning_rate = lr
    cfg.train.max_iters = max_iters
    cfg.train.eval_interval = eval_interval
    cfg.train.out_dir = out_dir

    model, tok, cfg = load_model_and_tokenizer(config_path, base_checkpoint)
    model = model.to(device)
    model.train()
    optimizer = model.configure_optimizers(cfg.train)

    # Tokenize instruction pairs.
    with open(sft_pairs_file, "r", encoding="utf-8") as f:
        raw = f.read()
    # Expect JSON list of [instruction, response] pairs.
    pairs = json.loads(raw)
    if not isinstance(pairs, list):
        raise ValueError("sft_pairs_file must contain a JSON list of [instruction, response]")
    pairs = [(str(a), str(b)) for a, b in pairs]
    tokenized = tokenize_pairs(pairs, tok, block_size)
    if not tokenized:
        raise ValueError("no usable instruction pairs after tokenization")

    # Shuffle deterministically.
    rng = np.random.RandomState(seed)
    order = list(range(len(tokenized)))
    rng.shuffle(order)

    log_path = os.path.join(out_dir, "sft-log.jsonl")
    start = time.perf_counter()
    total_tokens = 0
    best = float("inf")
    step = 0

    while step < max_iters:
        for idx in order:
            if step >= max_iters:
                break
            ids, mask = tokenized[idx]
            total_tokens += len(ids)
            x = torch.tensor(np.asarray(ids[:-1])[None, :], dtype=torch.long, device=device)
            y = torch.tensor(np.asarray(ids[1:])[None, :], dtype=torch.long, device=device)
            m = torch.tensor(np.asarray(mask[:-1])[None, :], dtype=torch.float32, device=device)
            logits, _ = model(x)
            loss = F_cross_entropy(logits, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % eval_interval == 0 or step == max_iters:
                vloss = sft_eval_loss(model, tokenized, device)
                log_entry = {"step": step, "sft_val_loss": round(vloss, 6)}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print(f"[sft] step {step}: val_loss {vloss:.4f}")
                if vloss < best:
                    best = vloss
                    torch.save({
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "stage": "sft",
                    }, os.path.join(out_dir, "sckpt-best.pt"))

    dt = time.perf_counter() - start
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "stage": "sft",
    }, os.path.join(out_dir, "sckpt-final.pt"))

    metrics = {
        "stage": "sft",
        "base_checkpoint": base_checkpoint,
        "n_pairs": len(pairs),
        "max_iters": max_iters,
        "elapsed_s": round(dt, 3),
        "tokens_per_sec": round(total_tokens / max(dt, 1e-9), 2),
        "best_sft_val_loss": round(best, 6),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "sft-metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[sft] done in {dt:.1f}s. best_sft_val_loss={best:.4f}")
    return metrics


def generate_response(model: HelixTransformer, tok: ByteLevelBPETokenizer,
                      instruction: str, device: str = "cpu",
                      max_new_tokens: int = 40, temperature: float = 0.7) -> str:
    """Generate a response to an instruction (response-only decoding)."""
    prompt = f"INSTRUCTION: {instruction}\nRESPONSE: "
    from helix.infer import generate
    out = generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                   temperature=temperature, top_k=50, device=device)
    # Strip the prompt echo.
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()
