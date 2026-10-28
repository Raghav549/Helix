"""Train and evaluate a from-scratch vision encoder aligned to HELIX's own LM.

This is deliberately a small, synthetic-domain caption experiment, not a
natural-image captioner. The transformer LM is initialized from a HELIX
checkpoint; the image encoder and projection are randomly initialized.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .image_understanding import ImageUnderstandingConfig, MultimodalHelix
from .runtime import configure_threads
from .tokenizer import ByteLevelBPETokenizer
from .visual import COLORS, POSITIONS, SHAPES, make_scenes

PROMPT = "Image: "


def _sequences(labels, tok):
    prefix = tok.encode(PROMPT)
    inputs, targets, captions = [], [], []
    for c, s, p in labels:
        caption = f"{COLORS[int(c)]} {SHAPES[int(s)]} {POSITIONS[int(p)]}"
        ids = prefix + tok.encode(caption) + [tok._id_for_bytes(b"<|endoftext|>")]
        inputs.append(ids[:-1])
        targets.append([-100] * (len(prefix) - 1) + ids[len(prefix):])
        captions.append(caption)
    return inputs, targets, captions


def _batch(images, inputs, targets, indices, tok):
    length = max(len(inputs[i]) for i in indices)
    x = torch.full((len(indices), length), tok._id_for_bytes(b"<|pad|>"), dtype=torch.long)
    y = torch.full((len(indices), length), -100, dtype=torch.long)
    for row, idx in enumerate(indices):
        x[row, :len(inputs[idx])] = torch.tensor(inputs[idx])
        y[row, :len(targets[idx])] = torch.tensor(targets[idx])
    pixels = torch.from_numpy(images[indices]).permute(0, 3, 1, 2).float() / 255
    return pixels, x, y


@torch.no_grad()
def _loss(model, images, inputs, targets, tok):
    model.eval(); total = 0.0; tokens = 0
    for start in range(0, len(images), 16):
        idx = list(range(start, min(start + 16, len(images))))
        pixels, x, y = _batch(images, inputs, targets, idx, tok)
        logits, _ = model(pixels, x)
        total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum'))
        tokens += int((y != -100).sum())
    return total / max(tokens, 1)


@torch.no_grad()
def describe_image(model, tok, image):
    """Greedy caption for a single RGB PIL image, with bounded context."""
    array = np.array(image.convert('RGB').resize((32, 32)), dtype=np.uint8)
    device = next(model.parameters()).device
    pixels = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255).to(device)
    prefix = tok.encode(PROMPT)
    model.eval()
    result = model.generate(pixels, torch.tensor([prefix], device=device), max_new_tokens=20,
                            temperature=1.0, top_k=1, device=str(device), eos_token_id=tok._id_for_bytes(b"<|endoftext|>"))
    generated = result[0, len(prefix):].tolist()
    eos = tok._id_for_bytes(b"<|endoftext|>")
    if eos in generated: generated = generated[:generated.index(eos)]
    return tok.decode(generated).strip()


def load_captioner(out_dir='artifacts/caption', device='cpu'):
    configure_threads(); root = Path(out_dir)
    cfg = json.loads((root / 'config.json').read_text())
    model = MultimodalHelix(ModelConfig(**cfg['llm']), ImageUnderstandingConfig(**cfg['vision']))
    checkpoint = torch.load(root / 'ckpt-best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model']); model.to(device).eval()
    tok = ByteLevelBPETokenizer.load(cfg['vocab'], cfg['merges'])
    return model, tok


def train_captioner(*, train_count=240, val_count=90, data_seed=1337,
                    base_checkpoint='artifacts/llm/ckpt-best.pt', base_config='configs/helix-research.json',
                    vocab='artifacts/data/vocab.json', merges='artifacts/data/merges.json', out_dir='artifacts/caption',
                    steps=800, batch_size=16, seed=42):
    if steps < 1 or batch_size < 1: raise ValueError('steps and batch_size must be positive')
    configure_threads(); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    start = time.monotonic(); root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(Path(base_config).read_text())
    llm_cfg = ModelConfig(**base_cfg['model'])
    vision_cfg = ImageUnderstandingConfig(image_size=32, patch_size=8, vit_dim=64, vit_layers=1,
                                          vit_heads=4, max_image_tokens=17)
    model = MultimodalHelix(llm_cfg, vision_cfg)
    checkpoint = torch.load(base_checkpoint, map_location='cpu', weights_only=True)
    model.llm.load_state_dict(checkpoint['model'])
    tok = ByteLevelBPETokenizer.load(vocab, merges)
    train_images, _, train_labels, train_manifest = make_scenes(train_count, data_seed)
    heldout_images, _, heldout_labels, val_manifest = make_scenes(val_count, data_seed + 100_000)
    images = torch.round((train_images.permute(0, 2, 3, 1) + 1) * 127.5).byte().numpy()
    val_images = torch.round((heldout_images.permute(0, 2, 3, 1) + 1) * 127.5).byte().numpy()
    import hashlib
    seen = {hashlib.sha256(a.tobytes()).digest() for a in images}
    if any(hashlib.sha256(a.tobytes()).digest() in seen for a in val_images):
        raise ValueError('exact image overlap between train and holdout')
    x, y, _ = _sequences(train_labels, tok); vx, vy, captions = _sequences(heldout_labels, tok)
    if not len(images) or not len(val_images): raise ValueError('both synthetic splits must be non-empty')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    config = {'llm': asdict(llm_cfg), 'vision': asdict(vision_cfg), 'vocab': str(vocab), 'merges': str(merges),
              'base_checkpoint': str(base_checkpoint), 'train_manifest': train_manifest, 'val_manifest': val_manifest,
              'steps': steps, 'batch_size': batch_size, 'seed': seed, 'task': 'synthetic color/shape/position captions'}
    (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    logs = []; best = float('inf')
    initial = _loss(model, val_images, vx, vy, tok)
    logs.append({'step': 0, 'val_loss': initial})
    for step in range(1, steps + 1):
        indices = torch.randint(len(images), (batch_size,), generator=generator).tolist()
        pixels, ix, iy = _batch(images, x, y, indices, tok)
        model.train(); optimizer.zero_grad(set_to_none=True)
        _, loss = model(pixels, ix, iy); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            value = _loss(model, val_images, vx, vy, tok)
            logs.append({'step': step, 'train_loss': loss.item(), 'val_loss': value})
            print(f'caption step {step}/{steps} train={loss.item():.4f} heldout={value:.4f}', flush=True)
            if value < best:
                best = value
                torch.save({'model': model.state_dict(), 'step': step, 'val_loss': value}, root / 'ckpt-best.pt')
    torch.save({'model': model.state_dict(), 'step': steps}, root / 'ckpt-final.pt')
    selected = torch.load(root / 'ckpt-best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model']); model.eval()
    from PIL import Image
    results = []
    for array, target in zip(val_images, captions):
        predicted = describe_image(model, tok, Image.fromarray(array))
        results.append({'target': target, 'prediction': predicted})
    def parts(caption): return caption.strip().lower().split()
    metrics = {'stage': 'synthetic_caption', 'initialization': 'random vision encoder + HELIX-trained language checkpoint',
               'parameters': sum(p.numel() for p in model.parameters()), 'base_checkpoint': str(base_checkpoint),
               'domain': 'HELIX-rendered primitives only; no natural photo validation',
               'seed': seed, 'train_examples': len(images), 'heldout_examples': len(val_images),
               'steps': steps, 'batch_size': batch_size, 'initial_heldout_loss': initial,
               'best_heldout_loss': best, 'selected_step': selected['step'],
               'exact_caption_accuracy': sum(parts(r['prediction']) == parts(r['target']) for r in results) / len(results),
               'color_accuracy': sum(bool(parts(r['prediction'])) and parts(r['prediction'])[0] == parts(r['target'])[0] for r in results) / len(results),
               'shape_accuracy': sum(len(parts(r['prediction'])) > 1 and parts(r['prediction'])[1] == parts(r['target'])[1] for r in results) / len(results),
               'position_accuracy': sum(len(parts(r['prediction'])) > 2 and parts(r['prediction'])[2] == parts(r['target'])[2] for r in results) / len(results),
               'elapsed_seconds': round(time.monotonic() - start, 2)}
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (root / 'eval.json').write_text(json.dumps(results, indent=2) + '\n')
    (root / 'train_log.json').write_text(json.dumps(logs, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
