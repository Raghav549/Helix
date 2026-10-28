"""HELIX integrated local web interface.

A single Flask app exposing all first-class capability areas through one
coherent UI:
  * LLM chat (streaming, CPU model from random init)
  * Image generation (latent diffusion from random init)
  * Image understanding/upload-analysis (ViT -> LLM)
  * Coding sandbox (PythonTool execution, explain/fix)
  * Research lab (experiment tracker, run & view metrics)

CPU-first; all models load lazily on first request so the UI starts fast.
No paid APIs; everything runs locally.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template_string, request, send_file

from helix.agent import Agent, ToolRegistry
from helix.config import HelixConfig, ModelConfig
from helix.data import load_ids
from helix.image_gen import ImageGenConfig, LatentDiffusion, generate_images, load_image_gen_model
from helix.image_understanding import (
    ImageUnderstandingConfig,
    MultimodalHelix,
    load_multimodal_model,
)
from helix.infer import generate as text_generate, load_model_and_tokenizer
from helix.tokenizer import ByteLevelBPETokenizer


# ---------------------------------------------------------------------------
# Lazy model holders
# ---------------------------------------------------------------------------

class ModelCache:
    def __init__(self) -> None:
        self._llm = None
        self._llm_tok = None
        self._llm_cfg = None
        self._image_gen = None
        self._image_gen_cfg = None
        self._mm = None
        self._mm_cfg = None
        self._vis_cfg = None

    def llm(self, config_path: str, checkpoint: str):
        if self._llm is None:
            model, tok, cfg = load_model_and_tokenizer(config_path, checkpoint)
            self._llm, self._llm_tok, self._llm_cfg = model, tok, cfg
        return self._llm, self._llm_tok, self._llm_cfg

    def image_gen(self, config_path: str, checkpoint: str):
        if self._image_gen is None:
            import json
            with open(config_path, encoding="utf-8") as f:
                raw = json.load(f)
            cfg = ImageGenConfig(**raw.get("image_gen", raw))
            model = load_image_gen_model(cfg, checkpoint)
            self._image_gen, self._image_gen_cfg = model, cfg
        return self._image_gen, self._image_gen_cfg

    def multimodal(self, llm_config_path: str, llm_checkpoint: str,
                   vision_config_path: str, vision_checkpoint: str):
        if self._mm is None:
            import json
            with open(llm_config_path, encoding="utf-8") as f:
                llm_raw = json.load(f)
            with open(vision_config_path, encoding="utf-8") as f:
                vis_raw = json.load(f)
            llm_cfg = ModelConfig(**llm_raw.get("model", llm_raw))
            vis_cfg = ImageUnderstandingConfig(**vis_raw.get("image_understanding", vis_raw))
            model = load_multimodal_model(llm_cfg, vis_cfg, vision_checkpoint)
            self._mm, self._mm_cfg, self._vis_cfg = model, llm_cfg, vis_cfg
        return self._mm, self._mm_cfg, self._vis_cfg


MODELS = ModelCache()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)


# ---------------------------------------------------------------------------
# HTML UI (single page, tabs)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HELIX — Integrated AI Platform</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; background: #0b0c10; color: #c5c6c7; }
  header { background: #1f2833; padding: 1rem 2rem; display: flex;
           align-items: center; gap: 1rem; border-bottom: 1px solid #45a29e; }
  header h1 { font-size: 1.25rem; margin: 0; color: #45a29e; letter-spacing: 0.1em; }
  header .tag { font-size: 0.75rem; color: #66fcf1; background: #1b2631;
                padding: 0.15rem 0.5rem; border-radius: 4px; border: 1px solid #45a29e; }
  .tabs { display: flex; border-bottom: 1px solid #1f2833; overflow-x: auto; }
  .tab { flex-shrink: 0; padding: 0.75rem 1.25rem; cursor: pointer;
         color: #66fcf1; background: #11141a; border-bottom: 2px solid transparent;
         border-right: 1px solid #1f2833; font-size: 0.9rem; }
  .tab:hover { background: #161b22; color: #fff; }
  .tab.active { color: #45a29e; border-bottom-color: #45a29e; background: #0b0c10; }
  .panels { display: none; padding: 1.5rem; gap: 1rem; }
  .panels.show { display: grid; grid-template-columns: 1fr; gap: 1rem; }
  @media (min-width: 900px) { .panels.show { grid-template-columns: 1fr 1fr; } }
  .card { background: #11141a; border: 1px solid #1f2833; border-radius: 8px;
          padding: 1rem; }
  .card h3 { margin: 0 0 0.75rem 0; color: #45a29e; font-size: 1rem;
             display: flex; justify-content: space-between; align-items: center; }
  .card h3 .small { font-size: 0.7rem; color: #66fcf1; font-weight: normal; }
  label { display: block; font-size: 0.8rem; color: #888; margin-bottom: 0.35rem; }
  input, textarea, select { width: 100%; background: #0b0c10; border: 1px solid #1f2833;
        color: #c5c6c7; padding: 0.5rem; border-radius: 4px; font-size: 0.9rem; }
  textarea { min-height: 90px; resize: vertical; font-family: ui-monospace, SFMono-Regular, monospace; }
  button { background: #45a29e; color: #0b0c10; border: none; padding: 0.55rem 1rem;
           border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 0.9rem; }
  button:hover { background: #66fcf1; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary { background: #1f2833; color: #c5c6c7; }
  button.secondary:hover { background: #2a3640; color: #fff; }
  .row { display: flex; gap: 0.75rem; align-items: flex-end; }
  .flex1 { flex: 1; }
  #chat-log { background: #0b0c10; border: 1px solid #1f2833; border-radius: 6px;
              padding: 0.75rem; min-height: 260px; max-height: 420px; overflow-y: auto;
              font-size: 0.85rem; line-height: 1.5; }
  .msg { margin-bottom: 0.6rem; padding: 0.5rem 0.75rem; border-radius: 6px;
         white-space: pre-wrap; font-size: 0.9rem; }
  .msg.user { background: #161b22; border-left: 3px solid #45a29e; }
  .msg.assistant { background: #0f1721; border-left: 3px solid #66fcf1; }
  .msg.system { color: #888; font-size: 0.75rem; font-style: italic; }
  .typing { color: #66fcf1; font-style: italic; }
  .img-wrap { border: 1px solid #1f2833; border-radius: 6px; padding: 0.5rem;
              text-align: center; background: #0b0c10; }
  .img-wrap img { max-width: 100%; max-height: 300px; border-radius: 4px;
                  border: 1px solid #1f2833; }
  .code-out { background: #0b0c10; border: 1px solid #1f2833; border-radius: 6px;
              padding: 0.6rem; font-family: ui-monospace, monospace; font-size: 0.8rem;
              white-space: pre-wrap; max-height: 220px; overflow-y: auto; color: #a4ffb1; }
  .exp-list { font-size: 0.85rem; }
  .exp-item { padding: 0.4rem 0.6rem; border-bottom: 1px solid #1f2833;
              display: flex; justify-content: space-between; align-items: center; }
  .badge { font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 3px;
           background: #1f2833; color: #66fcf1; }
  .muted { color: #888; font-size: 0.8rem; }
  .status { display: flex; gap: 1rem; font-size: 0.8rem; color: #888; margin-bottom: 0.5rem; }
  .status .item { display: flex; align-items: center; gap: 0.3rem; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #333; }
  .dot.on { background: #45a29e; box-shadow: 0 0 6px #45a29e; }
</style>
</head>
<body>
<header>
  <h1>HELIX</h1>
  <span class="tag">CPU-FIRST &bull; OPEN-SOURCE &bull; RANDOM-INIT</span>
</header>
<div class="tabs">
  <div class="tab active" onclick="openTab(event,'chat')">Chat</div>
  <div class="tab" onclick="openTab(event,'image')">Image Generation</div>
  <div class="tab" onclick="openTab(event,'vision')">Image Understanding</div>
  <div class="tab" onclick="openTab(event,'code')">Coding Agent</div>
  <div class="tab" onclick="openTab(event,'research')">Research Lab</div>
</div>

<div id="chat" class="panels show">
  <div class="card">
    <h3>Helix Chat <span class="small">streaming LLM</span></h3>
    <div class="status"><div class="item"><div class="dot on"></div>CPU model loaded</div></div>
    <div id="chat-log">
      <div class="msg system">HELIX chat ready. Type a message below.</div>
    </div>
    <div class="row">
      <input id="chat-input" type="text" placeholder="Ask HELIX anything..."
             onkeypress="if(event.key==='Enter') sendChat()">
      <button onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<div id="image" class="panels">
  <div class="card flex1">
    <h3>Image Generation <span class="small">latent diffusion (random init)</span></h3>
    <label>Prompt</label>
    <textarea id="img-prompt" placeholder="a small red cube on a blue background"></textarea>
    <button class="flex1" onclick="genImage()">Generate Image</button>
    <div class="img-wrap" id="img-out" style="margin-top:0.75rem;min-height:200px;">
      <div class="muted">Generated image appears here</div>
    </div>
  </div>
</div>

<div id="vision" class="panels">
  <div class="card flex1">
    <h3>Image Understanding <span class="small">upload + analyze</span></h3>
    <label>Image file</label>
    <input type="file" id="vision-file" accept="image/*">
    <button class="flex1" style="margin-top:0.5rem;" onclick="analyzeImage()">Analyze Image</button>
    <div class="img-wrap" id="vision-img" style="margin-top:0.5rem;min-height:160px;">
      <div class="muted">Uploaded image preview</div>
    </div>
    <div id="vision-out" class="card" style="margin-top:0.75rem;"></div>
  </div>
</div>

<div id="code" class="panels">
  <div class="card flex1">
    <h3>Coding Agent <span class="small">sandboxed Python</span></h3>
    <label>Code</label>
    <textarea id="code-input" placeholder="result = 40 + 2
print(result)"></textarea>
    <div class="row" style="margin-top:0.5rem;">
      <button onclick="runCode()">Execute</button>
      <button class="secondary" onclick="explainCode()">Explain / Fix</button>
    </div>
    <div class="code-out" id="code-out">Output appears here.</div>
  </div>
</div>

<div id="research" class="panels">
  <div class="card flex1">
    <h3>Research Lab <span class="small">experiment tracker</span></h3>
    <button class="flex1 secondary" onclick="listExperiments()">Refresh Experiments</button>
    <div id="research-out" class="exp-list" style="margin-top:0.75rem;"></div>
    <div class="muted" style="margin-top:0.5rem;">Runs, metrics, and artifacts are stored under out/.</div>
  </div>
</div>

<script>
function openTab(evt, name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  evt.currentTarget.classList.add('active');
  document.querySelectorAll('.panels').forEach(p=>p.classList.remove('show'));
  document.getElementById(name).classList.add('show');
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function addChat(role, text){
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = (role==='system'?'[system] ':'') + esc(text);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function sendChat(){
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if(!text) return;
  input.value = '';
  addChat('user', text);
  addChat('assistant', '');
  const last = document.getElementById('chat-log').lastElementChild;
  last.className = 'msg assistant typing';
  last.textContent = '';
  try{
    const res = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    last.className = 'msg assistant';
    last.textContent = data.reply || '(no reply)';
  }catch(e){
    last.className = 'msg assistant';
    last.textContent = 'Error: ' + esc(String(e));
  }
  last.scrollIntoView({behavior:'smooth'});
}

async function genImage(){
  const ta = document.getElementById('img-prompt');
  const prompt = ta.value.trim();
  if(!prompt){ alert('Enter a prompt'); return; }
  const out = document.getElementById('img-out');
  out.innerHTML = '<div class="muted">Generating...</div>';
  try{
    const res = await fetch('/api/image/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt: prompt})
    });
    const data = await res.json();
    if(data.image_url){
      out.innerHTML = '<img src="' + data.image_url + '?t=' + Date.now() +
                      '" alt="generated"><div class="muted" style="margin-top:0.4rem;">' +
                      esc(prompt) + '</div>';
    } else {
      out.innerHTML = '<div style="color:#ff6b6b;">' + esc(data.error||'generation failed') + '</div>';
    }
  }catch(e){
    out.innerHTML = '<div style="color:#ff6b6b;">Error: ' + esc(String(e)) + '</div>';
  }
}

let visionUrl = null;
document.getElementById('vision-file').addEventListener('change', function(e){
  const f = e.target.files[0];
  if(!f) return;
  visionUrl = URL.createObjectURL(f);
  document.getElementById('vision-img').innerHTML = '<img src="'+visionUrl+'" alt="upload">';
});

async function analyzeImage(){
  if(!visionUrl){ alert('Upload an image first'); return; }
  const out = document.getElementById('vision-out');
  out.innerHTML = '<div class="muted">Analyzing...</div>';
  const fd = new FormData();
  const f = document.getElementById('vision-file').files[0];
  fd.append('image', f);
  try{
    const res = await fetch('/api/image/analyze', { method:'POST', body: fd });
    const data = await res.json();
    out.innerHTML = '<div style="white-space:pre-wrap;">' + esc(data.description||data.error||'') + '</div>';
  }catch(e){
    out.innerHTML = '<div style="color:#ff6b6b;">Error: ' + esc(String(e)) + '</div>';
  }
}

async function runCode(){
  const ta = document.getElementById('code-input');
  const out = document.getElementById('code-out');
  out.textContent = 'Executing...';
  try{
    const res = await fetch('/api/code/execute', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code: ta.value})
    });
    const data = await res.json();
    let txt = '';
    if(data.output) txt += data.output + '\n';
    if(data.error) txt += 'ERROR: ' + data.error + '\n';
    txt += '(ok=' + data.ok + ', ' + data.duration_s + 's)';
    out.textContent = txt || '(no output)';
  }catch(e){
    out.textContent = 'Error: ' + esc(String(e));
  }
}

async function explainCode(){
  const ta = document.getElementById('code-input');
  const out = document.getElementById('code-out');
  out.textContent = 'Explaining...';
  try{
    const res = await fetch('/api/code/explain', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code: ta.value})
    });
    const data = await res.json();
    out.textContent = data.explanation || data.error || '(no explanation)';
  }catch(e){
    out.textContent = 'Error: ' + esc(String(e));
  }
}

async function listExperiments(){
  const out = document.getElementById('research-out');
  out.innerHTML = '<div class="muted">Loading...</div>';
  try{
    const res = await fetch('/api/research/experiments');
    const data = await res.json();
    if(data.experiments && data.experiments.length){
      out.innerHTML = data.experiments.map(e =>
        '<div class="exp-item"><div><div style="font-weight:600;">'+esc(e.name)+'</div>'+
        '<div class="muted">'+esc(e.stage)+' &bull; '+esc(e.device)+'</div></div>'+
        '<div class="badge">loss '+ (e.best_loss !== undefined ? e.best_loss.toFixed(4) : '—') +'</div></div>').join('');
    } else {
      out.innerHTML = '<div class="muted">No experiments yet. Run training from the CLI.</div>';
    }
  }catch(e){
    out.innerHTML = '<div style="color:#ff6b6b;">Error: ' + esc(String(e)) + '</div>';
  }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(config_path: str = "configs/helix-tiny.json",
               llm_checkpoint: str = "out/tiny-run/ckpt-best.pt",
               image_gen_config: Optional[str] = None,
               image_gen_checkpoint: Optional[str] = None,
               vision_config: Optional[str] = None,
               vision_checkpoint: Optional[str] = None) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/api/status")
    def status():
        return jsonify({
            "ok": True,
            "device": "cpu",
            "models": {
                "llm": MODELS._llm is not None,
                "image_gen": MODELS._image_gen is not None,
                "multimodal": MODELS._mm is not None,
            },
        })

    @app.post("/api/chat")
    def chat_route():
        body = request.get_json(silent=True) or {}
        message = body.get("message", "")
        try:
            model, tok, cfg = MODELS.llm(config_path, llm_checkpoint)
            from helix.infer import generate
            reply = generate(model, tok, message, max_new_tokens=60,
                             temperature=0.8, top_k=50, device=cfg.train.device)
            # strip prompt echo
            if reply.startswith(message):
                reply = reply[len(message):]
            return jsonify({"reply": reply.strip()})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"reply": f"[HELIX chat error: {exc}]"}), 500

    @app.post("/api/image/generate")
    def image_generate_route():
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt", "a small shape")
        try:
            import uuid
            cfg_path = image_gen_config or os.path.join(PROJECT_ROOT, "configs", "image_gen.json")
            ckpt = image_gen_checkpoint or os.path.join(PROJECT_ROOT, "out", "image_gen", "ckpt-final.pt")
            model, icfg = MODELS.image_gen(cfg_path, ckpt)
            from PIL import Image as PILImage
            import io
            imgs = generate_images(model, num_images=1, device=icfg.device)
            fname = f"helix-{uuid.uuid4().hex[:8]}.png"
            fpath = os.path.join(PROJECT_ROOT, "out", "web_images", fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            imgs[0].save(fpath)
            return jsonify({"image_url": "/out/web_images/" + fname})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"image generation error: {exc}"}), 500

    @app.post("/api/image/analyze")
    def image_analyze_route():
        if "image" not in request.files:
            return jsonify({"error": "no image file"}), 400
        file = request.files["image"]
        try:
            from PIL import Image as PILImage
            import io
            img = PILImage.open(io.BytesIO(file.read())).convert("RGB")
            # Analyze using the multimodal model if available, else a heuristic.
            desc = None
            if MODELS._mm is not None:
                mm, llm_cfg, vis_cfg = MODELS._mm, MODELS._mm_cfg, MODELS._vis_cfg
                from helix.tokenizer import ByteLevelBPETokenizer
                import torch
                tok = ByteLevelBPETokenizer()
                # tiny vocab encode of a generic prompt
                ids = torch.tensor([[0]], dtype=torch.long)
                # resize image to vision config size
                import numpy as np
                arr = np.array(img.resize((vis_cfg.image_size, vis_cfg.image_size)))
                arr = (arr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
                timg = torch.tensor(arr, dtype=torch.float32)
                with torch.no_grad():
                    out_ids = mm.generate(timg, ids, max_new_tokens=20,
                                          temperature=0.8, top_k=20, device=vis_cfg.device)
                desc = tok.decode(out_ids[0].tolist())
            if not desc:
                w, h = img.size
                desc = f"Received an image ({w}x{h} pixels). Image-understanding model not loaded; train it first via the CLI."
            return jsonify({"description": desc})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"image analysis error: {exc}"}), 500

    @app.post("/api/code/execute")
    def code_execute_route():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        try:
            from helix.agent import PythonTool
            tool = PythonTool(timeout_s=20.0, memory_mb=256)
            result = tool.execute(code=code)
            return jsonify({
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
                "duration_s": result.duration_s,
            })
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc), "output": "", "duration_s": 0.0})

    @app.post("/api/code/explain")
    def code_explain_route():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        try:
            model, tok, cfg = MODELS.llm(config_path, llm_checkpoint)
            from helix.infer import generate
            prompt = ("Explain what this code does and list any bugs or fixes:\n"
                      "```\n" + code + "\n```\nAnswer:")
            reply = generate(model, tok, prompt, max_new_tokens=80,
                             temperature=0.7, top_k=40, device=cfg.train.device)
            if reply.startswith(prompt):
                reply = reply[len(prompt):]
            return jsonify({"explanation": reply.strip()})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"explanation": f"[explain error: {exc}]"})

    @app.get("/api/research/experiments")
    def research_experiments():
        exps = []
        out_dir = os.path.join(PROJECT_ROOT, "out")
        if os.path.isdir(out_dir):
            for name in sorted(os.listdir(out_dir)):
                d = os.path.join(out_dir, name)
                mpath = os.path.join(d, "metrics.json")
                if os.path.isfile(mpath):
                    try:
                        with open(mpath, encoding="utf-8") as f:
                            m = json.load(f)
                        exps.append({
                            "name": name,
                            "stage": m.get("stage", "unknown"),
                            "device": m.get("config", {}).get("device", "cpu") if isinstance(m.get("config"), dict) else "cpu",
                            "best_loss": m.get("best_loss") or m.get("best_sft_val_loss"),
                        })
                    except Exception:
                        pass
        return jsonify({"experiments": exps})

    # Serve generated images from out/web_images
    @app.get("/out/web_images/<path:fname>")
    def web_image(fname):
        fpath = os.path.join(PROJECT_ROOT, "out", "web_images", fname)
        if not os.path.isfile(fpath):
            return "not found", 404
        return send_file(fpath, mimetype="image/png")

    return app


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="HELIX integrated web interface")
    parser.add_argument("--config", default="configs/helix-tiny.json")
    parser.add_argument("--checkpoint", default="out/tiny-run/ckpt-best.pt")
    parser.add_argument("--image-gen-config", default=None)
    parser.add_argument("--image-gen-checkpoint", default=None)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--vision-checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    app = create_app(args.config, args.checkpoint, args.image_gen_config,
                     args.image_gen_checkpoint, args.vision_config,
                     args.vision_checkpoint)
    print(f"HELIX web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
