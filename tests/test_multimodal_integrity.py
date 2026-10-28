"""Conditioning, synthetic rendering and captioning work without hidden pretrained weights."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from helix.config import ModelConfig
from helix.image_gen import ImageGenConfig, LatentDiffusion, load_image_gen_model
from helix.image_understanding import ImageUnderstandingConfig, MultimodalHelix
from helix.physics import Scene, render_scene, scene_from_prompt
from helix.visual import make_scenes

ROOT = Path(__file__).resolve().parents[1]


def test_renderer_geometry_light_and_deterministic_synthetic_split():
    scene = scene_from_prompt('a blue sphere on the right with light from the left', 64)
    assert (scene.color, scene.shape, scene.position, scene.light) == ('blue', 'sphere', 'right', 'left')
    left, a = render_scene(Scene('sphere', 'blue', 'left', 'matte', 'left', size=64))
    right, b = render_scene(Scene('sphere', 'blue', 'right', 'matte', 'right', size=64))
    assert a['object_pixels'] > 0 and a['shadow_pixels'] > 0
    assert not np.array_equal(np.asarray(left), np.asarray(right))
    one = make_scenes(8, 123); two = make_scenes(8, 123); heldout = make_scenes(8, 456)
    assert torch.equal(one[0], two[0]) and one[3]['pixels_sha256'] == two[3]['pixels_sha256']
    assert one[3]['pixels_sha256'] != heldout[3]['pixels_sha256']


def test_multimodal_respects_context_and_untied_head_receives_gradient():
    cfg = ModelConfig(vocab_size=40, block_size=12, n_layer=1, n_head=2, n_embd=32,
                      n_kv_head=1, dropout=0, weight_tying=False)
    vcfg = ImageUnderstandingConfig(image_size=16, patch_size=4, vit_dim=32,
                                    vit_layers=1, vit_heads=2, max_image_tokens=17)
    model = MultimodalHelix(cfg, vcfg)
    ids = torch.randint(0, 40, (2, 8)); image = torch.randn(2, 3, 16, 16)
    logits, loss = model(image, ids, torch.randint(0, 40, (2, 8)))
    assert logits.shape == (2, 8, 40)
    loss.backward()
    assert model.llm.lm_head.weight.grad is not None and model.vision.patch_embed.proj.weight.grad is not None
    model.eval()
    with torch.no_grad():
        first, _ = model(torch.zeros_like(image), ids)
        second, _ = model(torch.ones_like(image), ids)
    assert not torch.allclose(first, second)
    # With a long text prompt, generation reserves image tokens within the model window.
    assert model.generate(image[:1], ids[:1], max_new_tokens=8, top_k=1).shape == (1, 16)


@pytest.mark.skipif(not (ROOT / 'artifacts/image-gen/ckpt-final.pt').exists(), reason='trained artifacts absent')
def test_trained_diffusion_is_seeded_and_prompt_conditioned():
    cfg = ImageGenConfig(**json.loads((ROOT/'artifacts/image-gen/config.json').read_text())['image_gen'])
    model = load_image_gen_model(cfg, str(ROOT/'artifacts/image-gen/ckpt-final.pt')).eval()
    with torch.no_grad():
        a = model.sample(prompts=['red sphere on the left'], seed=17, timesteps=4)
        b = model.sample(prompts=['red sphere on the left'], seed=17, timesteps=4)
        c = model.sample(prompts=['blue cube on the right'], seed=17, timesteps=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)  # conditioning changes computation; not a quality claim


@pytest.mark.skipif(not (ROOT / 'artifacts/caption/ckpt-best.pt').exists(), reason='trained artifacts absent')
def test_real_captioner_is_image_conditioned_within_synthetic_domain():
    from helix.caption_workflow import load_captioner, describe_image
    model, tok = load_captioner(str(ROOT/'artifacts/caption'))
    red, _ = render_scene(Scene('sphere', 'red', 'left', size=32))
    blue, _ = render_scene(Scene('cube', 'blue', 'right', size=32))
    first = describe_image(model, tok, red); second = describe_image(model, tok, blue)
    assert first == 'red sphere left' and second == 'blue cube right'


def test_model_configs_reject_silently_unsupported_settings():
    with pytest.raises(ValueError): ImageGenConfig(unet_blocks=1)
    with pytest.raises(ValueError): ImageGenConfig(dtype='float16')
    with pytest.raises(ValueError): ImageUnderstandingConfig(image_size=32, patch_size=7)
    with pytest.raises(ValueError): ImageUnderstandingConfig(vit_dim=32, vit_heads=3)
