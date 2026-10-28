"""HELIX image understanding — ViT encoder + projector into the LLM embedding space.

Independently implemented from random initialization. No pretrained vision weights.

Architecture:
  * VisionTransformer: patch embedding + transformer encoder blocks
  * ImageProjection: linear projection from ViT dim -> LLM n_embd
  * MultimodalHelix: wraps HelixTransformer; encodes an image into a sequence of
    "image prefix embeddings" that are prepended to the text token embeddings so
    the language model can condition on visual input and generate grounded text.

This is the image-understanding half of a multimodal model. Trained from random
init; kept small so it trains on CPU. Image understanding at this scale is
prototype-level and honestly documented.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from helix.config import ModelConfig
from helix.model import HelixTransformer, RMSNorm


@dataclass
class ImageUnderstandingConfig:
    image_size: int = 32
    image_channels: int = 3
    patch_size: int = 4
    vit_dim: int = 64
    vit_layers: int = 2
    vit_heads: int = 4
    vit_mlp_dim: int = 128
    vit_dropout: float = 0.1
    max_image_tokens: int = 64
    seed: int = 1337
    device: str = "cpu"


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_ch: int, dim: int) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: ImageUnderstandingConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, config.image_channels, config.vit_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.vit_dim))
        self.drop = nn.Dropout(config.vit_dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.vit_dim,
                    nhead=config.vit_heads,
                    dim_feedforward=config.vit_mlp_dim,
                    dropout=config.vit_dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.vit_layers)
            ]
        )
        self.norm = RMSNorm(config.vit_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class ImageProjection(nn.Module):
    def __init__(self, vit_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, vit_out: torch.Tensor) -> torch.Tensor:
        return self.proj(vit_out)


class MultimodalHelix(nn.Module):
    def __init__(
        self,
        llm_config: ModelConfig,
        vision_config: ImageUnderstandingConfig,
    ) -> None:
        super().__init__()
        self.llm_config = llm_config
        self.vision_config = vision_config
        self.vision = VisionTransformer(vision_config)
        self.projector = ImageProjection(vision_config.vit_dim, llm_config.n_embd)
        self.llm = HelixTransformer(llm_config)

    def _rope(self, T: int, device: torch.device):
        return self.llm._rope(T, device)

    def forward(
        self,
        images: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = input_ids.device
        B, T = input_ids.shape
        text_emb = self.llm.tok_emb(input_ids)
        seq = text_emb
        if images is not None:
            vit_out = self.vision(images)
            img_emb = self.projector(vit_out)
            max_tok = min(self.vision_config.max_image_tokens, img_emb.size(1))
            img_emb = img_emb[:, :max_tok, :]
            seq = torch.cat([img_emb, text_emb], dim=1)
        x = self.llm.drop(seq)
        total_T = seq.size(1)
        cos, sin = self._rope(total_T, device)
        if self.llm.config.grad_checkpoint and self.llm.training:
            from torch.utils.checkpoint import checkpoint

            for block in self.llm.blocks:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
        else:
            for block in self.llm.blocks:
                x = block(x, cos, sin)
        x = self.llm.norm_f(x)
        if images is not None:
            text_start = min(self.vision_config.max_image_tokens, img_emb.size(1))
        else:
            text_start = 0
        text_h = x[:, text_start:, :]
        logits = F.linear(text_h, self.llm.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        self.eval()
        img_emb = None
        if images is not None:
            vit_out = self.vision(images)
            img_emb = self.projector(vit_out)
            max_tok = min(self.vision_config.max_image_tokens, img_emb.size(1))
            img_emb = img_emb[:, :max_tok, :]
        idx = input_ids.clone()
        for _ in range(max_new_tokens):
            text_emb = self.llm.tok_emb(idx)
            if img_emb is not None:
                seq = torch.cat([img_emb.expand(idx.size(0), -1, -1), text_emb], dim=1)
            else:
                seq = text_emb
            x = self.llm.drop(seq)
            total_T = seq.size(1)
            cos, sin = self._rope(total_T, device)
            for block in self.llm.blocks:
                x = block(x, cos, sin)
            x = self.llm.norm_f(x)
            text_start = img_emb.size(1) if img_emb is not None else 0
            text_h = x[:, text_start:, :]
            logits = F.linear(text_h, self.llm.tok_emb.weight)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < v[:, [-1]], float("-inf"), logits)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_remove = cumulative_probs > top_p
                sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                sorted_remove[:, 0] = 0
                remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                logits = logits.masked_fill(remove, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def train_image_understanding(
    config: ImageUnderstandingConfig,
    llm_config: ModelConfig,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    targets: torch.Tensor,
    out_dir: str = "out/image_understanding",
    max_iters: int = 500,
    batch_size: int = 4,
    lr: float = 1e-3,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    import numpy as np
    np.random.seed(config.seed)
    device = config.device
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = MultimodalHelix(llm_config, config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    N = images.size(0)
    log_path = os.path.join(out_dir, "log.jsonl")
    start = time.perf_counter()
    best = float("inf")
    model.train()
    for it in range(max_iters):
        idx = torch.randint(0, N, (batch_size,))
        batch_img = images[idx].to(device)
        batch_txt = text_ids[idx].to(device)
        batch_tgt = targets[idx].to(device)
        opt.zero_grad(set_to_none=True)
        logits, loss = model(batch_img, batch_txt, batch_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == max_iters - 1:
            entry = {"iter": it, "loss": round(float(loss.item()), 6)}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[image_understanding] iter {it}: loss {loss.item():.4f}")
            if float(loss.item()) < best:
                best = float(loss.item())
                torch.save(model.state_dict(), os.path.join(out_dir, "ckpt-best.pt"))
    dt = time.perf_counter() - start
    torch.save(model.state_dict(), os.path.join(out_dir, "ckpt-final.pt"))
    metrics = {
        "stage": "image_understanding",
        "config": asdict(config),
        "llm_config": asdict(llm_config),
        "num_pairs": N,
        "max_iters": max_iters,
        "elapsed_s": round(dt, 3),
        "best_loss": round(best, 6),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[image_understanding] done in {dt:.1f}s. best_loss={best:.4f}")
    return metrics


def load_multimodal_model(
    llm_config: ModelConfig,
    vision_config: ImageUnderstandingConfig,
    ckpt_path: str,
    device: str = "cpu",
):
    model = MultimodalHelix(llm_config, vision_config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
