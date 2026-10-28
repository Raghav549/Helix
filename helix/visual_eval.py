"""Controlled comparison of learned-only vs renderer-guided image synthesis.

Reference-guided img2img has extra information; pixel agreement is a diagnostic,
not a fair text-to-image quality contest or a realism score. The classifier is
trained only on renderer images and may be unreliable on generated images.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from helix.image_gen import ImageGenConfig, load_image_gen_model, generate_images
from helix.physics import Scene, render_scene
from helix.runtime import configure_threads
from helix.visual import COLORS, POSITIONS, SHAPES, inspect_image, load_visual


def compare_physics(config_path: str, image_checkpoint: str, vision_checkpoint: str,
                    output_path: str, seed: int = 2026) -> dict:
    configure_threads()
    cfg = ImageGenConfig(**json.loads(Path(config_path).read_text())['image_gen'])
    if not cfg.use_text_conditioning:
        raise ValueError('comparison requires a text-conditioned model')
    model = load_image_gen_model(cfg, image_checkpoint, cfg.device)
    vision = load_visual(vision_checkpoint)
    stats = {name: {'mse_sum': 0.0, 'color': 0, 'shape': 0, 'position': 0, 'all': 0}
             for name in ('learned_only', 'physics_guided', 'renderer_reference')}
    cases = []
    for color in COLORS:
        for shape in SHAPES:
            for position in POSITIONS:
                i = len(cases)
                scene = Scene(shape, color, position, 'matte', 'left', 45., cfg.image_size)
                prompt = f'A matte {color} {shape} on the {position}, light from the left.'
                reference, meta = render_scene(scene)
                arr = np.asarray(reference, dtype=np.float32)
                initial = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
                plain = generate_images(model, prompts=[prompt], seed=seed+i, device=cfg.device)[0]
                guided = generate_images(model, prompts=[prompt], seed=seed+i, init_images=initial,
                                         strength=.25, device=cfg.device)[0]
                for mode, image in [('learned_only', plain), ('physics_guided', guided),
                                    ('renderer_reference', reference)]:
                    mse = float(np.mean(((np.asarray(image, dtype=np.float32)-arr)/255.)**2))
                    pred = inspect_image(image, vision)['synthetic_domain_predictions']
                    matches = [pred[k]['label'] == v for k, v in [('color', color), ('shape', shape), ('position', position)]]
                    stats[mode]['mse_sum'] += mse
                    for key, ok in zip(('color', 'shape', 'position'), matches): stats[mode][key] += int(ok)
                    stats[mode]['all'] += int(all(matches))
                cases.append({'prompt': prompt, 'seed': seed+i, 'renderer_shadow_pixels': meta['shadow_pixels']})
    count = len(cases)
    summary = {}
    for mode, values in stats.items():
        summary[mode] = {'mean_pixel_mse_to_renderer': round(values['mse_sum']/count, 6),
                         **{k + '_accuracy': round(values[k]/count, 4) for k in ('color', 'shape', 'position', 'all')}}
    report = {'cases': count, 'seed': seed, 'model_checkpoint': image_checkpoint,
              'vision_checkpoint': vision_checkpoint, 'guided_strength': .25,
              'results': summary, 'prompts': cases,
              'limitations': ['The guided model receives the reference image; learned-only does not.',
                              'Pixel agreement to the renderer is not physical realism.',
                              'Attribute classifier is validated on synthetic renders only, not generated images.',
                              '18 fixed synthetic scenes; no natural-photo or human evaluation.']}
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
