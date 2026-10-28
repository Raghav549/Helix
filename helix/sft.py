"""Supervised instruction tuning (SFT) — a stage strictly separate from pretraining.

Pretraining (helix/train.py) optimizes next-token prediction on a reviewed corpus.
SFT (this module) optimizes instruction responses: given an instruction, produce
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
from pathlib import Path
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

    # Fallback synthetic pairs from declarative sentences. These are deterministic
    # auditable prompts for the authored smoke corpus, not claimed domain knowledge.
    for sent in re.split(r"[.!?]+", text):
        sent = _clean(sent)
        if len(sent) < 12:
            continue
        low = sent.lower()
        if any(w in low for w in (" is ", " are ", " means ", " was ", " were ")):
            topic = re.sub(r"^the |^a |^an ", "", sent.split(" is ")[0].strip())[:48]
            instr = f"Review this authored claim about {topic}:" if topic else f"Review this authored claim: {sent[:40]}"
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
    if block_size < 2:
        raise ValueError("SFT block_size must be at least 2")
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for instr, resp in pairs:
        # Encode separately: BPE merges spanning the prefix/response boundary
        # would otherwise invalidate the loss mask. EOS is a reserved token.
        prefix_ids = tok.encode(f"INSTRUCTION: {instr}\nRESPONSE: ")
        response_ids = tok.encode(resp, add_eos=True)
        if not prefix_ids or not response_ids:
            continue
        minimum_response = min(len(response_ids), max(1, block_size // 2))
        prefix_ids = prefix_ids[-max(1, block_size + 1 - minimum_response):]
        response_ids = response_ids[:block_size + 1 - len(prefix_ids)]
        ids = prefix_ids + response_ids
        mask = np.zeros(len(ids), dtype=np.float32)
        mask[len(prefix_ids)-1:len(ids)-1] = 1.0
        if mask.sum():
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
              seed: int = 1337, device: str = "cpu", patience: int = 4,
              min_delta: float = 1e-4) -> dict:
    """SFT on a deterministic train/holdout split, with response-only loss."""
    import hashlib
    from helix.runtime import configure_threads
    from helix.storage import resolve_data_root
    if min(max_iters, eval_interval, batch_size, patience) < 1 or min_delta < 0:
        raise ValueError("iterations, interval, batch size and patience must be positive; min_delta cannot be negative")
    # All experiment artifacts, including temporary instruction data, live in
    # the configured external root; no temporary data is written to the repo.
    temporary_dir = resolve_data_root(os.environ.get('HELIX_DATA_ROOT')) / 'artifacts/sft-smoke'
    temporary_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = temporary_dir / Path(sft_pairs_file).name
    if not Path(sft_pairs_file).is_absolute():
        pairs_path = temporary_dir / sft_pairs_file
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(Path(sft_pairs_file).read_text(encoding='utf-8'), encoding='utf-8')
    sft_pairs_file = str(pairs_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    configure_threads()
    os.makedirs(out_dir, exist_ok=True)
    model, tok, cfg = load_model_and_tokenizer(config_path, base_checkpoint)
    block_size = min(block_size, model.config.block_size)
    cfg.train.device, cfg.train.block_size = device, block_size
    cfg.train.batch_size, cfg.train.learning_rate = batch_size, lr
    cfg.train.max_iters, cfg.train.eval_interval, cfg.train.out_dir = max_iters, eval_interval, out_dir
    model.to(device).train()
    optimizer = model.configure_optimizers(cfg.train)
    with open(sft_pairs_file, "r", encoding="utf-8") as f:
        raw = f.read()
    pairs = json.loads(raw)
    if not isinstance(pairs, list):
        raise ValueError("sft_pairs_file must contain a JSON list of pairs")
    pairs = [(str(a), str(b)) for a, b in pairs]
    examples = tokenize_pairs(pairs, tok, block_size)
    if not examples:
        raise ValueError("no usable instruction pairs after tokenization")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(examples)).tolist()
    val_count = max(1, len(examples) // 10) if len(examples) > 1 else 0
    validation = [examples[i] for i in order[:val_count]]
    training = [examples[i] for i in order[val_count:]]
    pad_id = tok._id_for_bytes(b"<|pad|>")
    log_path = os.path.join(out_dir, "sft-log.jsonl")
    initial_val_loss = sft_eval_loss(model, validation, device) if validation else None
    initial_train_loss = sft_eval_loss(model, training, device)
    start = time.perf_counter()
    total_tokens = 0
    best = float("inf")
    bad_evals = 0
    actual_iters = 0
    early_stopped = False
    for step in range(1, max_iters + 1):
        actual_iters = step
        selected = [training[int(i)] for i in rng.randint(0, len(training), size=batch_size)]
        length = max(len(ids) - 1 for ids, _ in selected)
        x = torch.full((batch_size, length), pad_id, dtype=torch.long, device=device)
        y = x.clone()
        mask = torch.zeros((batch_size, length), dtype=torch.float32, device=device)
        for row, (ids, m) in enumerate(selected):
            n = len(ids) - 1
            x[row, :n] = torch.as_tensor(ids[:-1], device=device)
            y[row, :n] = torch.as_tensor(ids[1:], device=device)
            mask[row, :n] = torch.as_tensor(m[:-1], device=device)
            total_tokens += n
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F_cross_entropy(logits, y, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        if step % eval_interval == 0 or step == max_iters:
            train_loss = sft_eval_loss(model, training, device)
            val_loss = sft_eval_loss(model, validation, device) if validation else None
            selection_loss = val_loss if val_loss is not None else train_loss
            log_entry = {"step": step, "train_loss": train_loss, "sft_val_loss": val_loss}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[sft] step {step}: train_loss {train_loss:.4f} " +
                  (f"held_out_loss {val_loss:.4f}" if val_loss is not None else "no held-out pair"))
            if selection_loss < best - min_delta:
                best = selection_loss
                bad_evals = 0
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "stage": "sft"},
                           os.path.join(out_dir, "sckpt-best.pt"))
            else:
                bad_evals += 1
                if validation and bad_evals >= patience and step < max_iters:
                    early_stopped = True
                    print(f"[sft] early stopping at step {step}: held-out loss did not improve for {patience} evaluations")
                    break
    dt = time.perf_counter() - start
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "stage": "sft"},
               os.path.join(out_dir, "sckpt-final.pt"))
    metrics = {"stage": "sft", "base_checkpoint": base_checkpoint,
               "pairs_sha256": hashlib.sha256(raw.encode()).hexdigest(),
               "n_pairs": len(pairs), "train_pairs": len(training), "heldout_pairs": len(validation),
               "config": asdict(cfg), "seed": seed,
               "initial_train_loss": initial_train_loss, "initial_heldout_loss": initial_val_loss,
               "final_train_loss": sft_eval_loss(model, training, device),
               "final_heldout_loss": sft_eval_loss(model, validation, device) if validation else None,
               "max_iters": max_iters, "actual_iters": actual_iters,
               "early_stopped": early_stopped, "patience": patience, "min_delta": min_delta,
               "elapsed_s": round(dt, 3),
               "tokens_processed": total_tokens, "tokens_per_sec": round(total_tokens / max(dt, 1e-9), 2),
               "best_sft_val_loss": round(best, 6) if validation else None,
               "checkpoint_selection": "held-out loss" if validation else "training loss (no holdout)",
               "out_dir": out_dir}
    with open(os.path.join(out_dir, "sft-metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[sft] done in {dt:.1f}s")
    return metrics


def generate_response(model: HelixTransformer, tok: ByteLevelBPETokenizer,
                      instruction: str, device: str = "cpu",
                      max_new_tokens: int = 40, temperature: float = 0.7) -> str:
    """Generate a response to an instruction (response-only decoding)."""
    prompt = f"INSTRUCTION: {instruction}\nRESPONSE: "
    from helix.infer import stream_generate
    return "".join(stream_generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=50, device=device)).strip()
