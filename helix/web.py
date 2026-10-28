"""Local integrated HELIX application. No paid APIs or pretrained models.

The bundled models are small research prototypes, not production assistants.
Code execution is explicitly approved and fails closed without Linux isolation.
The server has no authentication: bind to loopback except in trusted previews.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

from helix.memory import MemoryStore
from helix.storage import resolve_data_root

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)


def _path(value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)


class ModelCache:
    def __init__(self):
        self._llm = self._image = self._vision = self._caption = None
        self._lock = threading.Lock()

    def llm(self, config, checkpoint):
        with self._lock:
            if self._llm is None:
                from helix.infer import load_model_and_tokenizer
                self._llm = load_model_and_tokenizer(config, checkpoint)
            return self._llm

    def image(self, config, checkpoint):
        with self._lock:
            if self._image is None:
                from helix.image_gen import ImageGenConfig, load_image_gen_model
                raw = json.loads(Path(config).read_text(encoding='utf-8'))
                cfg = ImageGenConfig(**raw.get('image_gen', raw))
                self._image = (load_image_gen_model(cfg, checkpoint, cfg.device), cfg)
            return self._image

    def vision(self, checkpoint):
        with self._lock:
            if self._vision is None:
                from helix.visual import load_visual
                self._vision = load_visual(checkpoint)
            return self._vision

    def caption(self, directory):
        with self._lock:
            if self._caption is None:
                from helix.caption_workflow import load_captioner
                self._caption = load_captioner(directory)
            return self._caption


def _json_body():
    try:
        body = request.get_json(silent=True)
    except (RecursionError, ValueError) as exc:
        raise ValueError('invalid JSON') from exc
    if not isinstance(body, dict):
        raise ValueError('a JSON object is required')
    # JSON can represent escaped lone surrogates and Python's decoder also
    # accepts NaN/Infinity. Reject them before logging, tokenization or SQLite.
    import math
    stack = [(body, 0)]; count = 0
    while stack:
        value, depth = stack.pop(); count += 1
        if count > 100_000 or depth > 64:
            raise ValueError('JSON structure exceeds limits')
        if isinstance(value, str):
            try: value.encode('utf-8')
            except UnicodeEncodeError as exc: raise ValueError('JSON strings must be valid UTF-8') from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        elif isinstance(value, dict):
            stack.extend((v, depth + 1) for v in value.values())
            stack.extend((k, depth + 1) for k in value.keys())
        elif isinstance(value, list):
            stack.extend((v, depth + 1) for v in value)
    return body


def _read_image(upload):
    raw = upload.read(6 * 1024 * 1024 + 1)
    if len(raw) > 6 * 1024 * 1024:
        raise ValueError('image exceeds 6 MiB')
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 16_000_000 or min(image.size) < 1:
            raise ValueError('image resolution outside limits')
        image.load()
        return ImageOps.exif_transpose(image).convert('RGB')


def _origin_parts(value: str):
    """Parse a serialized HTTP(S) origin; reject paths, userinfo and bad ports."""
    if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                parsed.username is not None or parsed.password is not None or
                parsed.path or '?' in value or '#' in value):
            return None
        _ = parsed.port  # Accessing this validates malformed/out-of-range ports.
        return parsed.scheme, parsed.netloc.lower()
    except ValueError:
        return None


def create_app(config_path: str = 'configs/helix-research.json',
               llm_checkpoint: str = 'artifacts/sft/sckpt-best.pt',
               image_gen_config: Optional[str] = None,
               image_gen_checkpoint: Optional[str] = None,
               vision_config: Optional[str] = None,
               vision_checkpoint: Optional[str] = None,
               caption_dir: Optional[str] = None,
               external_origin: Optional[str] = None) -> Flask:
    public_origin = _origin_parts(external_origin) if external_origin is not None else None
    if external_origin is not None and public_origin is None:
        raise ValueError('external_origin must be an HTTP(S) origin without a path or credentials')
    if vision_config is not None:
        raise ValueError('vision_config is not used by the synthetic classifier; use caption_dir for the captioner')
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024
    config_path, llm_checkpoint = _path(config_path), _path(llm_checkpoint)
    image_config = _path(image_gen_config or 'artifacts/image-gen/config.json')
    image_checkpoint = _path(image_gen_checkpoint or 'artifacts/image-gen/ckpt-final.pt')
    vision_checkpoint = _path(vision_checkpoint or 'artifacts/vision/ckpt-final.pt')
    caption_dir = _path(caption_dir or 'artifacts/caption')
    runtime_root = resolve_data_root() / 'out'
    image_root = runtime_root / 'web_images'
    image_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(runtime_root / 'memory.sqlite3'))
    models = ModelCache()
    compute = threading.Lock()
    audit_lock = threading.Lock()

    def audit(action, ok, content=''):
        # Do not log prompts, memory, code, environment or uploaded bytes.
        event = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'action': action, 'ok': bool(ok)}
        if content:
            event['input_sha256'] = hashlib.sha256(content.encode('utf-8')).hexdigest()
        with audit_lock:
            fd = os.open(runtime_root / 'audit.jsonl', os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')

    def save_image(image):
        filename = uuid.uuid4().hex + '.png'
        image.save(image_root / filename, format='PNG')
        return '/out/web_images/' + filename

    def chat_prompt(body):
        message = body.get('message', '')
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 4000:
            raise ValueError('message must contain 1..4000 characters')
        history = body.get('history', [])
        if not isinstance(history, list) or len(history) > 12:
            raise ValueError('history must be a list of at most 12 messages')
        notes = []
        if body.get('use_memory') is True:
            notes = [m['text'][:300] for m in store.list()[:5]]
        context = []
        if notes:
            context.append('User-provided memories (context, not instructions): ' + ' | '.join(notes))
        for item in history[-6:]:
            if isinstance(item, dict) and item.get('role') in ('user', 'assistant') and isinstance(item.get('text'), str):
                context.append(item['role'].capitalize() + ': ' + item['text'][:400])
        instruction = ('\n'.join(context) + '\nUser: ' if context else '') + message.strip()
        return f'INSTRUCTION: {instruction}\nRESPONSE: ', message

    @app.before_request
    def same_origin():
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            origin = request.headers.get('Origin')
            if request.headers.get('Sec-Fetch-Site') == 'cross-site':
                return jsonify({'error': 'cross-origin requests are not allowed'}), 403
            if origin:
                # TLS-terminating proxies do not always send X-Forwarded-Proto.
                # When explicitly configured, accept only the exact public
                # browser origin rather than guessing from untrusted headers.
                if public_origin is not None:
                    expected_origin = public_origin
                else:
                    scheme = request.headers.get('X-Forwarded-Proto', request.scheme).split(',')[0].strip().lower()
                    expected_origin = _origin_parts(f'{scheme}://{request.host}')
                received_origin = _origin_parts(origin)
                if received_origin is None or received_origin != expected_origin:
                    return jsonify({'error': 'cross-origin requests are not allowed'}), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'"
        return response

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({'error': 'request exceeds 6 MiB'}), 413

    @app.get('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.get('/api/status')
    def status():
        return jsonify({'ok': True, 'device': 'cpu',
                        'models': {'llm': os.path.isfile(config_path) and os.path.isfile(llm_checkpoint),
                                   'image_gen': os.path.isfile(image_config) and os.path.isfile(image_checkpoint),
                                   'vision': os.path.isfile(vision_checkpoint),
                                   'caption': os.path.isfile(os.path.join(caption_dir, 'ckpt-best.pt'))},
                        'loaded': {'llm': models._llm is not None, 'image_gen': models._image is not None,
                                   'vision': models._vision is not None, 'caption': models._caption is not None},
                        'limitations': 'Small experimental models. No general-purpose accuracy or photorealism claim.'})

    def start_chat(body, streaming):
        try:
            prompt, message = chat_prompt(body)
            limit = int(body.get('max_new_tokens', 80))
            if not 1 <= limit <= 160: raise ValueError('max_new_tokens must be 1..160')
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        if not compute.acquire(blocking=False):
            return jsonify({'error': 'another compute task is running; try again'}), 429
        try:
            model, tok, cfg = models.llm(config_path, llm_checkpoint)
        except Exception as exc:
            compute.release()
            audit('chat', False, message)
            return jsonify({'error': f'language checkpoint unavailable: {exc}'}), 503
        from helix.infer import stream_generate
        if streaming:
            def events():
                try:
                    for chunk in stream_generate(model, tok, prompt, limit, temperature=.7,
                                                 top_k=40, device=cfg.train.device):
                        yield 'data: ' + json.dumps({'delta': chunk}, ensure_ascii=False) + '\n\n'
                    audit('chat', True, message)
                    yield 'data: {"done":true}\n\n'
                except Exception as exc:
                    audit('chat', False, message)
                    yield 'data: ' + json.dumps({'error': str(exc)}) + '\n\n'
                finally:
                    compute.release()
            return Response(events(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        try:
            reply = ''.join(stream_generate(model, tok, prompt, limit, temperature=.7, top_k=40,
                                            device=cfg.train.device))
            audit('chat', True, message)
            return jsonify({'reply': reply, 'verified': False,
                            'notice': 'Prototype language-model output; verify independently.'})
        except Exception as exc:
            audit('chat', False, message)
            return jsonify({'error': str(exc)}), 500
        finally:
            compute.release()

    @app.post('/api/chat')
    def chat_route():
        try: return start_chat(_json_body(), False)
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.post('/api/chat/stream')
    def chat_stream():
        try: return start_chat(_json_body(), True)
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.route('/api/memory', methods=['GET', 'POST', 'DELETE'])
    def memory_route():
        if request.method == 'GET':
            return jsonify({'memories': store.list(), 'encrypted': False})
        if request.method == 'DELETE':
            count = store.clear(); audit('memory_clear', True)
            return jsonify({'deleted': count})
        try:
            body = _json_body()
            memory_id = store.add(body.get('text', ''), consent=body.get('consent') is True)
            audit('memory_add', True)
            return jsonify({'id': memory_id}), 201
        except (ValueError, PermissionError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/memory/<int:memory_id>', methods=['PUT', 'DELETE'])
    def memory_item(memory_id):
        try:
            if request.method == 'DELETE':
                ok = store.delete(memory_id)
            else:
                body = _json_body()
                ok = store.update(memory_id, body.get('text', ''), consent=body.get('consent') is True)
            audit('memory_change', ok)
            return jsonify({'ok': ok}), 200 if ok else 404
        except (ValueError, PermissionError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/image/generate')
    def image_generate():
        try:
            body = _json_body()
            prompt = body.get('prompt', '')
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 1024:
                raise ValueError('prompt must contain 1..1024 UTF-8 bytes')
            mode = body.get('mode', 'physics')
            if mode not in ('physics', 'learned', 'physics_guided'):
                raise ValueError('unknown image mode')
            seed = int(body.get('seed', 1337))
            if not 0 <= seed < 2**32: raise ValueError('seed must be between 0 and 2^32-1')
            if mode != 'learned' and not re.search(r'\b(sphere|cube|box)\b', prompt.lower()):
                raise ValueError('physics scene mode supports a sphere or cube; include one in the prompt')
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        if not compute.acquire(blocking=False):
            return jsonify({'error': 'another compute task is running; try again'}), 429
        try:
            from helix.physics import scene_from_prompt, render_scene
            metadata = {}
            if mode == 'physics':
                image, metadata = render_scene(scene_from_prompt(prompt, 256))
                notice = 'Controlled direct-light renderer; supports one primitive, not general text-to-image.'
            else:
                import numpy as np
                import torch
                from helix.image_gen import generate_images
                model, cfg = models.image(image_config, image_checkpoint)
                if not cfg.use_text_conditioning:
                    raise ValueError('image checkpoint does not support prompts')
                init = None
                if mode == 'physics_guided':
                    reference, metadata = render_scene(scene_from_prompt(prompt, cfg.image_size))
                    arr = np.asarray(reference, dtype=np.float32).copy()
                    init = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
                    metadata['reference_url'] = save_image(reference)
                image = generate_images(model, prompts=[prompt], seed=seed, init_images=init,
                                        strength=.25, device=cfg.device)[0]
                notice = ('Experimental 32×32 diffusion; prompt fidelity and realism are limited.'
                          if mode == 'learned' else
                          'Renderer-guided image-to-image: the model receives a rendered reference; output is experimental.')
            url = save_image(image)
            audit('image_generate_' + mode, True, prompt)
            return jsonify({'image_url': url, 'mode': mode, 'seed': seed,
                            'notice': notice, 'metadata': metadata})
        except Exception as exc:
            audit('image_generate_' + mode, False, prompt)
            return jsonify({'error': str(exc)}), 500
        finally:
            compute.release()

    @app.post('/api/image/analyze')
    def image_analyze():
        upload = request.files.get('image')
        if upload is None: return jsonify({'error': 'image file required'}), 400
        if not compute.acquire(blocking=False):
            return jsonify({'error': 'another compute task is running; try again'}), 429
        try:
            from helix.visual import inspect_image
            image = _read_image(upload)
            model = models.vision(vision_checkpoint) if os.path.isfile(vision_checkpoint) else None
            result = inspect_image(image, model)
            if os.path.isfile(os.path.join(caption_dir, 'ckpt-best.pt')):
                from helix.caption_workflow import describe_image
                captioner, tokenizer = models.caption(caption_dir)
                result['synthetic_domain_caption'] = describe_image(captioner, tokenizer, image)
                result['caption_warning'] = 'Learned captioner trained only on synthetic primitives; not a natural-image description.'
            audit('image_analyze', True)
            return jsonify(result)
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            audit('image_analyze', False)
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            audit('image_analyze', False)
            return jsonify({'error': str(exc)}), 500
        finally:
            compute.release()

    @app.post('/api/image/edit')
    def image_edit():
        upload = request.files.get('image')
        if upload is None: return jsonify({'error': 'image file required'}), 400
        prompt = request.form.get('prompt', '')
        if not prompt.strip() or len(prompt.encode('utf-8')) > 1024: return jsonify({'error': 'prompt required (1..1024 UTF-8 bytes)'}), 400
        try:
            strength = float(request.form.get('strength', '.5'))
            seed = int(request.form.get('seed', '1337'))
            if not 0 < strength <= 1 or not 0 <= seed < 2**32: raise ValueError('invalid strength or seed')
            image = _read_image(upload)
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            return jsonify({'error': str(exc)}), 400
        if not compute.acquire(blocking=False): return jsonify({'error': 'another compute task is running'}), 429
        try:
            import numpy as np
            import torch
            from helix.image_gen import generate_images
            model, cfg = models.image(image_config, image_checkpoint)
            if not cfg.use_text_conditioning: raise ValueError('checkpoint does not support prompts')
            arr = np.asarray(image.resize((cfg.image_size, cfg.image_size)), dtype=np.float32).copy()
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
            edited = generate_images(model, prompts=[prompt], seed=seed, init_images=tensor,
                                     strength=strength, device=cfg.device)[0]
            audit('image_edit', True, prompt)
            return jsonify({'image_url': save_image(edited), 'seed': seed,
                            'notice': 'Experimental 32×32 image-to-image; not validated on photographs.'})
        except Exception as exc:
            audit('image_edit', False, prompt)
            return jsonify({'error': str(exc)}), 500
        finally:
            compute.release()

    @app.post('/api/code/analyze')
    def code_analyze():
        try:
            body = _json_body(); code = body.get('code', '')
            if not isinstance(code, str) or len(code.encode()) > 128 * 1024: raise ValueError('code too large')
            from helix.coding import StaticAnalyzer
            result = asdict(StaticAnalyzer(code, '<workspace>').analyze())
            audit('code_analyze', True, code)
            return jsonify(result)
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.post('/api/code/execute')
    def code_execute():
        try:
            body = _json_body()
            if body.get('approved') is not True:
                return jsonify({'ok': False, 'error': 'explicit approval required'}), 403
            code = body.get('code', '')
            from helix.sandbox import run_python
            result = run_python(code, timeout_s=10, memory_mb=256)
            audit('code_execute', result.ok, code if isinstance(code, str) else '')
            return jsonify(asdict(result))
        except (ValueError, TypeError) as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

    @app.post('/api/code/patch')
    def code_patch():
        try:
            body = _json_body(); original, modified = body.get('original'), body.get('modified')
            if not isinstance(original, str) or not isinstance(modified, str) or len(original)+len(modified) > 128*1024:
                raise ValueError('original and modified source required, under 128 KiB combined')
            from helix.coding import PatchGenerator
            patch = PatchGenerator().make_patch('workspace.py', original, modified)
            audit('code_patch_preview', True)
            return jsonify({'diff': patch.diff, 'note': 'Preview only; no file was changed.'})
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.post('/api/code/test-skeleton')
    def code_test_skeleton():
        try:
            body = _json_body(); code = body.get('code', '')
            if not isinstance(code, str) or len(code.encode('utf-8')) > 128 * 1024:
                raise ValueError('Python source must be under 128 KiB')
            from helix.coding import TestGenerator
            tests = TestGenerator().generate_tests('workspace.py', code)
            audit('code_test_skeleton', True)
            return jsonify({'tests': tests, 'notice': 'Generated callable/import checks only; not executed or behaviorally verified.'})
        except (ValueError, SyntaxError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/code/explain')
    def code_explain():
        try:
            body = _json_body(); code = body.get('code', '')
            if not isinstance(code, str) or len(code) > 4000: raise ValueError('code must be under 4000 characters')
            return start_chat({'message': 'Explain this Python code:\n' + code, 'max_new_tokens': 100}, False)
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.post('/api/research/math')
    def research_math():
        try:
            body = _json_body()
            expression = body.get('expression', '')
            if not isinstance(expression, str) or len(expression) > 512: raise ValueError('expression must be under 512 characters')
            from helix.research import run_bounded_math, ExperimentTracker
            if not compute.acquire(blocking=False): return jsonify({'error': 'another compute task is running'}), 429
            try:
                result = run_bounded_math(body.get('operation', 'differentiate'), expression,
                                          body.get('variable', 'x'), body.get('point', '0'))
            finally:
                compute.release()
            if body.get('save_experiment') is True:
                tracker = ExperimentTracker(str(runtime_root / 'experiments'))
                eid = tracker.create('Symbolic calculation', 'research', {'operation': body.get('operation')},
                                     {'expression': expression, 'variable': body.get('variable', 'x')})
                tracker.set_outputs(eid, result)
                if result.get('ok'): tracker.add_verifier(eid, 'sympy')
                result['experiment_id'] = eid
            audit('research_math', result.get('ok'), expression)
            return jsonify(result), 200 if result.get('ok') else 400
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/research/csv')
    def research_csv():
        try:
            body = _json_body(); text = body.get('csv', '')
            if not isinstance(text, str) or len(text.encode()) > 1024*1024: raise ValueError('CSV must be under 1 MiB')
            from helix.research import summarize_csv
            result = summarize_csv(text)
            audit('research_csv', 'error' not in result)
            return jsonify(result), 400 if 'error' in result else 200
        except ValueError as exc: return jsonify({'error': str(exc)}), 400

    @app.post('/api/research/plot')
    def research_plot():
        try:
            body = _json_body(); text = body.get('csv', '')
            x_name, y_name = body.get('x'), body.get('y')
            if not isinstance(x_name, str) or not isinstance(y_name, str) or max(len(x_name), len(y_name)) > 128:
                raise ValueError('x and y column names required, at most 128 characters')
            from helix.research import load_csv, plot_line
            header, rows = load_csv(text)
            if x_name not in header or y_name not in header:
                raise ValueError('selected columns not found')
            if len(rows) > 5000:
                raise ValueError('plots support at most 5,000 rows')
            xi, yi = header.index(x_name), header.index(y_name)
            x = [row[xi] if xi < len(row) else '' for row in rows]
            y = [row[yi] if yi < len(row) else '' for row in rows]
            if not compute.acquire(blocking=False): return jsonify({'error': 'another compute task is running'}), 429
            try:
                filename = uuid.uuid4().hex + '.png'
                result = plot_line(x, y, f'{y_name} by {x_name}', str(image_root / filename), x_name, y_name)
            finally:
                compute.release()
            if 'error' in result:
                return jsonify(result), 400
            audit('research_plot', True)
            return jsonify({'image_url': '/out/web_images/' + filename, 'points': result['points'],
                            'note': 'Only rows with finite numeric values in both columns are plotted.'})
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/research/experiments')
    def research_experiments():
        entries = []
        for root in (resolve_data_root() / 'artifacts', runtime_root, runtime_root / 'experiments'):
            if not root.is_dir(): continue
            for directory in sorted(root.iterdir()):
                if not directory.is_dir() or directory.is_symlink(): continue
                for filename in ('record.json', 'metrics.json', 'sft-metrics.json'):
                    path = directory / filename
                    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000: continue
                    try:
                        data = json.loads(path.read_text(encoding='utf-8'))
                        entries.append({'name': data.get('name', directory.name),
                                        'stage': data.get('stage', 'unknown'),
                                        'device': data.get('device', 'cpu'),
                                        'metrics': data.get('heldout', data.get('metrics', {})),
                                        'best_loss': data.get('best_validation_loss', data.get('best_sft_val_loss', data.get('best_loss', data.get('best_heldout_loss'))))})
                    except (OSError, ValueError): continue
        return jsonify({'experiments': entries})

    @app.get('/out/web_images/<name>')
    def web_image(name):
        if not re.fullmatch(r'[0-9a-f]{32}\.png', name): return 'not found', 404
        return send_from_directory(image_root, name, mimetype='image/png')

    return app


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='HELIX local integrated web interface')
    parser.add_argument('--config', default='configs/helix-research.json')
    parser.add_argument('--checkpoint', default='artifacts/sft/sckpt-best.pt')
    parser.add_argument('--image-gen-config', default=None)
    parser.add_argument('--image-gen-checkpoint', default=None)
    parser.add_argument('--vision-config', default=None)
    parser.add_argument('--vision-checkpoint', default=None)
    parser.add_argument('--caption-dir', default=None)
    parser.add_argument('--external-origin', default=None, help='exact browser origin when behind a TLS-terminating proxy')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    app = create_app(args.config, args.checkpoint, args.image_gen_config,
                     args.image_gen_checkpoint, args.vision_config, args.vision_checkpoint, args.caption_dir,
                     external_origin=args.external_origin)
    print(f'HELIX local research interface on http://{args.host}:{args.port}; no authentication')
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
