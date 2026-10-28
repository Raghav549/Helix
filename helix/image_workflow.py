"""Integrated image-data review, training, evaluation and inference."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from helix.image_gen import ImageGenConfig, LatentDiffusion, train_image_gen, load_image_gen_model, generate_images
from helix.runtime import configure_threads
from helix.visual import make_scenes


def load_image_manifest(path: str, size: int = 32):
    """Ingest local images only with explicit rights and safety attestations.

    Manifest records require path, sha256, caption, license, license_evidence,
    rights_confirmed and safety_reviewed. This records human assertions; it is
    not an automatic legal or content-safety judgment. Rejected items are listed.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    records = data.get('images', [])
    if not isinstance(records, list) or len(records) > 10000:
        raise ValueError('manifest images must be a list of at most 10000 records')
    images, captions, accepted, rejected, seen, hashes = [], [], [], [], set(), set()
    for record in records:
        if not isinstance(record, dict):
            rejected.append({'path': '', 'reason': 'manifest record must be an object'})
            continue
        name = str(record.get('path', ''))
        try:
            if not (record.get('rights_confirmed') is True and record.get('safety_reviewed') is True
                    and isinstance(record.get('license'), str)
                    and record['license'].strip().lower() not in ('', 'unknown', 'varies', 'none')
                    and isinstance(record.get('license_evidence'), str) and record['license_evidence'].strip()
                    and isinstance(record.get('sha256'), str) and record['sha256']):
                raise ValueError('rights/licence/safety evidence not confirmed')
            caption = record.get('caption')
            if not isinstance(caption, str) or len(caption.strip()) < 3 or len(caption.encode()) > 1024:
                raise ValueError('caption must be 3..1024 UTF-8 bytes')
            image_path = (Path(path).resolve().parent / name).resolve()
            if not image_path.is_file() or image_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('missing image or file exceeds 16 MiB')
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest != record['sha256']:
                raise ValueError('sha256 mismatch')
            if digest in seen:
                raise ValueError('exact duplicate')
            with Image.open(image_path) as im:
                if min(im.size) < 16 or im.width * im.height > 16_000_000:
                    raise ValueError('image resolution outside limits')
                im = ImageOps.exif_transpose(im).convert('RGB')
                gray = np.asarray(im.resize((8, 8)).convert('L'))
                ahash = tuple((gray > gray.mean()).reshape(-1).tolist())
                if ahash in hashes:
                    raise ValueError('near-duplicate average hash')
                array = np.asarray(im.resize((size, size)), dtype=np.uint8).copy()
            seen.add(digest); hashes.add(ahash)
            images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1)
            captions.append(caption.strip())
            accepted.append({'path': name, 'sha256': digest, 'caption': caption,
                             'license': record['license'], 'license_evidence': record['license_evidence'],
                             'source_version': record.get('source_version'),
                             'attribution': record.get('attribution')})
        except (OSError, ValueError, UnidentifiedImageError, TypeError) as exc:
            rejected.append({'path': name, 'reason': str(exc)})
    if not images:
        raise ValueError('no eligible images; check manifest rights, safety, hashes, captions and files')
    return torch.stack(images), captions, {'accepted': accepted, 'quarantined': rejected,
        'note': 'Rights and safety are human attestations, not automated determinations.'}


@torch.no_grad()
def diffusion_loss(model, images, captions, seed):
    torch.manual_seed(seed)
    model.eval()
    # Fixed seed ensures random-init and trained comparisons use identical noise.
    out = model(images, prompts=captions if model.config.use_text_conditioning else None)
    return {key: round(float(out[key]), 6) for key in ('diff_loss', 'recon_loss', 'kl_loss')}


def train_synthetic_image_pipeline(config_path: str, out_dir: str,
                                   train_count: int = 240, val_count: int = 60) -> dict:
    configure_threads()
    raw = json.loads(Path(config_path).read_text(encoding='utf-8'))
    config = ImageGenConfig(**raw.get('image_gen', raw))
    train_x, train_captions, _, train_manifest = make_scenes(train_count, config.seed, config.image_size)
    val_x, val_captions, _, val_manifest = make_scenes(val_count, config.seed + 100_000, config.image_size)
    torch.manual_seed(config.seed)
    random_model = LatentDiffusion(config).to(config.device)
    before = diffusion_loss(random_model, val_x.to(config.device), val_captions, config.seed + 3)
    del random_model
    metrics = train_image_gen(config, train_x, out_dir=out_dir, captions=train_captions)
    trained = load_image_gen_model(config, os.path.join(out_dir, 'ckpt-final.pt'), config.device)
    after = diffusion_loss(trained, val_x.to(config.device), val_captions, config.seed + 3)
    metrics.update({'train_manifest': train_manifest, 'heldout_manifest': val_manifest,
                    'heldout_random_init': before, 'heldout_trained': after,
                    'heldout_note': 'fixed noise seed; held-out synthetic scenes, not a perceptual or prompt-adherence score'})
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, 'config.json').write_text(json.dumps({'image_gen': asdict(config)}, indent=2), encoding='utf-8')
    Path(out_dir, 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('[image_gen] held-out random:', before, 'trained:', after)
    return metrics


def generate_to_file(config_path: str, checkpoint: str, prompt: str, output_path: str,
                     seed: int = 1337, init_image: str | None = None, strength: float = .6):
    config = ImageGenConfig(**json.loads(Path(config_path).read_text(encoding='utf-8')).get('image_gen', {}))
    model = load_image_gen_model(config, checkpoint, config.device)
    source = None
    if init_image:
        with Image.open(init_image) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize((config.image_size, config.image_size))
            array = np.asarray(image, dtype=np.float32).copy()
        source = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    imgs = generate_images(model, prompts=[prompt] if config.use_text_conditioning else None,
                           seed=seed, init_images=source, strength=strength, device=config.device)
    if not config.use_text_conditioning and prompt:
        raise ValueError('cannot use a prompt with an unconditional checkpoint')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imgs[0].save(output_path)
    return {'path': output_path, 'seed': seed, 'size': imgs[0].size, 'prompt': prompt,
            'model': 'HELIX independently trained latent diffusion',
            'quality': 'experimental; no claim of photorealism or prompt fidelity'}
