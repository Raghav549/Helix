"""HELIX command-line interface.

Run ``python -m helix --help`` to see training, inference, evaluation,
image/vision, research, coding, corpus and local web commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from helix.train import train as train_fn


def cmd_train(args: argparse.Namespace) -> int:
    train_fn(args.config)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    if args.quantize:
        from helix.infer import load_quantized_model_and_tokenizer, generate_quantized

        model, tok, cfg = load_quantized_model_and_tokenizer(args.config, args.checkpoint)
        out = generate_quantized(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=cfg.train.device,
            use_cache=args.kv_cache,
        )
    elif args.kv_cache:
        from helix.infer import load_model_and_tokenizer, generate_with_cache

        model, tok, cfg = load_model_and_tokenizer(args.config, args.checkpoint)
        out = generate_with_cache(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=cfg.train.device,
        )
    else:
        from helix.infer import load_model_and_tokenizer, generate

        model, tok, cfg = load_model_and_tokenizer(args.config, args.checkpoint)
        out = generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=cfg.train.device,
        )
    print(out)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from helix.infer import chat

    chat(args.config, args.checkpoint)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import math

    import numpy as np
    import torch

    from helix.config import load_config
    from helix.data import get_batch, load_ids
    from helix.infer import load_model_and_tokenizer

    cfg = load_config(args.config)
    model, tok, cfg = load_model_and_tokenizer(args.config, args.checkpoint)
    if args.eval_iters < 1:
        raise ValueError("eval-iters must be positive")
    train_ids, val_ids = load_ids(cfg.data.train_file), load_ids(cfg.data.val_file)
    batch_rng = torch.Generator().manual_seed(args.seed)
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_iters):
            X, Y = get_batch(
                "val",
                train_ids,
                val_ids,
                cfg.train.batch_size,
                cfg.train.block_size,
                cfg.train.device,
                generator=batch_rng,
            )
            _, loss = model(X, Y)
            losses.append(loss.item())
    avg = float(np.mean(losses))
    print(f"eval_iters={args.eval_iters} val_loss={avg:.4f} perplexity={math.exp(avg):.2f}")

    # Run the original, contamination-resistant evaluation tasks and write a
    # JSON report alongside the perplexity number.
    tasks = None
    if args.eval_tasks:
        tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    try:
        from helix.eval import run_eval
        import json
        torch.manual_seed(args.seed)
        report = run_eval(args.config, args.checkpoint, args.out_path, tasks=tasks,
                          device=cfg.train.device)
        summary = report.summary()
        summary["validation"] = {"loss": avg, "perplexity": math.exp(avg),
                                 "batches": args.eval_iters, "batch_size": cfg.train.batch_size,
                                 "block_size": cfg.train.block_size, "seed": args.seed,
                                 "note": "sampled held-out token windows; not a general capability benchmark"}
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] report: {summary['passed']}/{summary['total']} "
              f"tasks passed -> {args.out_path}")
    except Exception as exc:  # pragma: no cover - errors must not masquerade as success
        print(f"[eval] task-suite failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json

    from helix.infer import load_model_and_tokenizer, generate

    model, tok, cfg = load_model_and_tokenizer(args.config, args.checkpoint)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return
            text = payload.get("prompt", "")
            out = generate(
                model,
                tok,
                text,
                max_new_tokens=int(payload.get("max_new_tokens", 50)),
                temperature=float(payload.get("temperature", 0.8)),
                top_k=payload.get("top_k"),
                top_p=payload.get("top_p"),
                device=cfg.train.device,
            )
            resp = json.dumps({"completion": out}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, fmt, *args):
            return  # silence default request logging

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"HELIX serving on http://127.0.0.1:{args.port} (POST /generate)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from helix.bench import run_bench

    run_bench(args.config, args.out_path,
              block_sizes=[int(s) for s in args.block_sizes.split(",") if s.strip()],
              checkpoint_path=args.checkpoint)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from helix.ablate import run_ablation

    run_ablation(args.config, args.out_path,
                 block_types=[t.strip() for t in args.block_types.split(",") if t.strip()])
    return 0


def cmd_sft(args: argparse.Namespace) -> int:
    from helix.sft import train_sft

    train_sft(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        sft_pairs_file=args.sft_pairs,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device="cpu",
    )
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from helix.agent import Agent, ToolRegistry, PythonTool, CalculatorTool

    from helix.infer import load_model_and_tokenizer

    model, tok, cfg = load_model_and_tokenizer(args.config, args.checkpoint)
    reg = ToolRegistry.default_registry()
    agent = Agent(model, tok, registry=reg, device=cfg.train.device,
                  max_steps=args.max_steps, max_time_s=args.max_time,
                  require_approval=True)
    run = agent.run(args.goal)
    print(f"Agent goal: {run.goal}")
    print(f"Stopped: {run.stopped_reason}  steps={len(run.steps)}  "
          f"time={run.total_duration_s}s")
    print(f"Final answer: {run.final_answer}")
    print("\nAudit log:")
    print(agent.audit_log_json())
    os.makedirs(os.path.dirname(args.audit_log) or ".", exist_ok=True)
    with open(args.audit_log, "w", encoding="utf-8") as f:
        f.write(agent.audit_log_json())
    print(f"\nAudit log written to {args.audit_log}")
    return 0



def cmd_sources(args: argparse.Namespace) -> int:
    from helix.pipeline import run
    try:
        result = run(root=args.data_root)
    except Exception as exc:
        print(f'HELIX source pipeline failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from helix.web import main as web_main
    # web_main parses its own args; rebuild argv for it.
    import sys as _sys
    argv = []
    if args.config: argv += ["--config", args.config]
    if args.checkpoint: argv += ["--checkpoint", args.checkpoint]
    if args.image_gen_config: argv += ["--image-gen-config", args.image_gen_config]
    if args.image_gen_checkpoint: argv += ["--image-gen-checkpoint", args.image_gen_checkpoint]
    if args.vision_config: argv += ["--vision-config", args.vision_config]
    if args.vision_checkpoint: argv += ["--vision-checkpoint", args.vision_checkpoint]
    if args.caption_dir: argv += ["--caption-dir", args.caption_dir]
    if args.external_origin is not None: argv += ["--external-origin", args.external_origin]
    argv += ["--host", args.host, "--port", str(args.port)]
    _sys.argv = ["helix"] + argv  # drop consumed "web" token
    return web_main()



def cmd_doctor(args):
    import json
    import importlib.metadata as metadata
    from helix.bench import detect_hardware
    from helix.sandbox import run_python
    packages = {}
    for name in ("torch", "numpy", "Pillow", "Flask", "sympy", "matplotlib", "pypdf"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    probe = run_python("result = 6 * 7", timeout_s=5)
    print(json.dumps({"hardware": detect_hardware().to_dict(), "packages": packages,
                      "sandbox_available": probe.ok, "sandbox_error": probe.error}, indent=2))
    return 0 if probe.ok else 1


def cmd_corpus_scan(args):
    import json
    from helix.corpus import scan_directory, RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE
    rights = []
    if args.allow_training: rights.append(RIGHTS_TRAIN)
    if args.allow_redistribution: rights.append(RIGHTS_REDISTRIBUTE)
    heldout = ()
    if args.heldout:
        with open(args.heldout, encoding="utf-8") as f:
            heldout = tuple(line.strip() for line in f if line.strip())
    m = scan_directory(args.directory, rights=rights, default_license=args.license,
                       license_evidence=args.evidence, source_version=args.version,
                       attribution=args.attribution, contamination_strings=heldout)
    m.save(args.manifest)
    print(json.dumps(m.summary(), indent=2))
    return 0


def cmd_corpus_build(args):
    import json
    from helix.corpus import CorpusManifest, build_training_corpus
    print(json.dumps(build_training_corpus(CorpusManifest.load(args.manifest), args.output), indent=2))
    return 0


def cmd_corpus_search(args):
    import json
    from helix.corpus import discover_github
    m = discover_github(args.query, args.limit)
    m.save(args.manifest)
    print(json.dumps(m.summary(), indent=2))
    print("All search results are quarantined pending per-file rights review.")
    return 0


def cmd_image_train(args):
    import json
    from dataclasses import asdict
    import torch
    from helix.image_gen import ImageGenConfig, train_image_gen, load_image_gen_model
    from helix.image_workflow import train_synthetic_image_pipeline, load_image_manifest, diffusion_loss
    if args.synthetic:
        train_synthetic_image_pipeline(args.config, args.out_dir, args.count, args.val_count)
        return 0
    raw = json.load(open(args.config, encoding="utf-8"))
    config = ImageGenConfig(**raw.get("image_gen", raw))
    if not config.use_text_conditioning:
        raise ValueError("image manifest training requires a text-conditioned config")
    images, captions, report = load_image_manifest(args.manifest, config.image_size)
    if len(images) < 2:
        raise ValueError("at least two reviewed images are required for a holdout")
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(config.seed)).tolist()
    holdout = max(1, len(images) // 10)
    val_i, train_i = order[:holdout], order[holdout:]
    train_caps, val_caps = [captions[i] for i in train_i], [captions[i] for i in val_i]
    metrics = train_image_gen(config, images[train_i], out_dir=args.out_dir, captions=train_caps)
    model = load_image_gen_model(config, os.path.join(args.out_dir, "ckpt-final.pt"), config.device)
    metrics["heldout_loss"] = diffusion_loss(model, images[val_i].to(config.device), val_caps, config.seed + 3)
    metrics["data_review"] = report
    metrics["train_records"] = train_i
    metrics["heldout_records"] = val_i
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"image_gen": asdict(config)}, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["heldout_loss"], indent=2))
    return 0


def cmd_image_generate(args):
    import json
    from helix.image_workflow import generate_to_file
    print(json.dumps(generate_to_file(args.config, args.checkpoint, args.prompt, args.output,
                                     args.seed, args.init_image, args.strength), indent=2))
    return 0


def cmd_scene(args):
    import json
    from helix.physics import scene_from_prompt, render_scene
    im, info = render_scene(scene_from_prompt(args.prompt, args.size))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    im.save(args.output)
    print(json.dumps({"path": args.output, **info}, indent=2))
    return 0


def cmd_vision_train(args):
    from helix.visual import train_visual
    train_visual(args.out_dir, train_count=args.count, val_count=args.val_count,
                 steps=args.steps, batch_size=args.batch_size, seed=args.seed)
    return 0


def cmd_image_analyze(args):
    import json
    from PIL import Image
    from helix.visual import load_visual, inspect_image
    model = load_visual(args.checkpoint) if args.checkpoint else None
    with Image.open(args.image) as im:
        result = inspect_image(im, model)
        if args.caption:
            from helix.caption_workflow import load_captioner, describe_image
            captioner, tokenizer = load_captioner(args.caption_dir)
            result["synthetic_domain_caption"] = describe_image(captioner, tokenizer, im)
            result["caption_warning"] = "Trained on synthetic primitives only; not a natural-image description."
    print(json.dumps(result, indent=2))
    return 0


def cmd_caption_train(args):
    from helix.caption_workflow import train_captioner
    train_captioner(steps=args.steps, batch_size=args.batch_size, out_dir=args.out_dir,
                    base_checkpoint=args.base_checkpoint)
    return 0


def cmd_visual_eval(args):
    import json
    from helix.visual_eval import compare_physics
    report = compare_physics(args.config, args.image_checkpoint, args.vision_checkpoint,
                             args.out_path, args.seed)
    print(json.dumps(report["results"], indent=2))
    return 0


def cmd_research(args):
    import json
    from helix.research import run_bounded_math
    result = run_bounded_math(args.operation, args.expression, args.variable, args.point)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def cmd_code_run(args):
    import json
    from helix.sandbox import run_python
    if not args.approve:
        print("Execution requires --approve", file=sys.stderr)
        return 2
    with open(args.file, encoding="utf-8") as f:
        code = f.read()
    from dataclasses import asdict
    result = run_python(code, timeout_s=args.timeout)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helix", description="HELIX CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train from random initialization")
    p_train.add_argument("--config", required=True, help="config JSON path")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="generate text from a prompt")
    p_gen.add_argument("--config", default="configs/helix-research.json")
    p_gen.add_argument("--checkpoint", default="artifacts/sft/sckpt-best.pt")
    p_gen.add_argument("--prompt", default="The helix is")
    p_gen.add_argument("--max-new-tokens", type=int, default=50)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k", type=int, default=None)
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.add_argument(
        "--kv-cache",
        action="store_true",
        help="use KV-cached incremental generation (benchmark vs full recompute)",
    )
    p_gen.add_argument(
        "--quantize",
        action="store_true",
        help="quantize Linear weights to int8 before generation (benchmark vs float32)",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_chat = sub.add_parser("chat", help="interactive chat")
    p_chat.add_argument("--config", default="configs/helix-research.json")
    p_chat.add_argument("--checkpoint", default="artifacts/sft/sckpt-best.pt")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("eval", help="validation loss / perplexity")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--eval-iters", type=int, default=50)
    p_eval.add_argument("--seed", type=int, default=1337)
    p_eval.add_argument("--out-path", default="out/eval-report.json",
                        help="path to write the JSON evaluation report")
    p_eval.add_argument("--eval-tasks", default=None,
                        help="comma-separated eval task names (default: all)")
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="local HTTP generation server")
    p_serve.add_argument("--config", required=True)
    p_serve.add_argument("--checkpoint", required=True)
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=cmd_serve)

    p_bench = sub.add_parser("bench", help="CPU benchmark: throughput, latency, RAM, scaling")
    p_bench.add_argument("--config", default="configs/helix-tiny.json")
    p_bench.add_argument("--out-path", default="out/bench-report.json")
    p_bench.add_argument("--checkpoint", default=None, help="trained checkpoint (default: random weights)")
    p_bench.add_argument("--block-sizes", default="64,128,256",
                         help="comma-separated context lengths to benchmark")
    p_bench.set_defaults(func=cmd_bench)

    p_ablate = sub.add_parser("ablate", help="architecture ablation: compare block types from random init")
    p_ablate.add_argument("--config", default="configs/helix-tiny.json")
    p_ablate.add_argument("--out-path", default="out/ablation-report.json")
    p_ablate.add_argument("--block-types", default="transformer,linear_attn,recurrent",
                          help="comma-separated block types to compare")
    p_ablate.set_defaults(func=cmd_ablate)

    p_sft = sub.add_parser("sft", help="supervised instruction tuning (separate stage from pretraining)")
    p_sft.add_argument("--config", required=True)
    p_sft.add_argument("--base-checkpoint", required=True)
    p_sft.add_argument("--sft-pairs", required=True, help="JSON list of [instruction, response] pairs")
    p_sft.add_argument("--out-dir", default="out/sft-run")
    p_sft.add_argument("--max-iters", type=int, default=100)
    p_sft.add_argument("--block-size", type=int, default=128)
    p_sft.add_argument("--batch-size", type=int, default=4)
    p_sft.add_argument("--lr", type=float, default=1e-4)
    p_sft.add_argument("--patience", type=int, default=4, help="SFT evaluation checks without improvement before early stopping")
    p_sft.set_defaults(func=cmd_sft)

    p_agent = sub.add_parser("agent", help="run a modular agent with sandboxed tools")
    p_agent.add_argument("--config", required=True)
    p_agent.add_argument("--checkpoint", required=True)
    p_agent.add_argument("--goal", required=True, help="task goal for the agent")
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.add_argument("--max-time", type=float, default=60.0)
    p_agent.add_argument("--audit-log", default="out/agent-audit.json",
                         help="path to write the audit log")
    p_agent.set_defaults(func=cmd_agent)

    p_sources = sub.add_parser("sources", help="run the external rights-aware corpus pipeline")
    p_sources.add_argument("--data-root", default=None)
    p_sources.set_defaults(func=cmd_sources)

    p_web = sub.add_parser("web", help="local integrated web UI (chat, images, code, research)")
    p_web.add_argument("--config", default="configs/helix-research.json")
    p_web.add_argument("--checkpoint", default="artifacts/sft/sckpt-best.pt")
    p_web.add_argument("--image-gen-config", default=None)
    p_web.add_argument("--image-gen-checkpoint", default=None)
    p_web.add_argument("--vision-config", default=None, help="deprecated; rejected (caption config is in --caption-dir)")
    p_web.add_argument("--vision-checkpoint", default=None)
    p_web.add_argument("--caption-dir", default=None)
    p_web.add_argument("--external-origin", default=None, help="exact browser origin behind a TLS-terminating proxy")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8766)
    p_web.set_defaults(func=cmd_web)

    p_doctor = sub.add_parser("doctor", help="check hardware, dependencies and execution isolation")
    p_doctor.set_defaults(func=cmd_doctor)

    p_scan = sub.add_parser("corpus-scan", help="review and scan a local source directory")
    p_scan.add_argument("--directory", required=True)
    p_scan.add_argument("--manifest", required=True)
    p_scan.add_argument("--license", default="unknown")
    p_scan.add_argument("--evidence", default=None)
    p_scan.add_argument("--version", default=None)
    p_scan.add_argument("--attribution", default=None)
    p_scan.add_argument("--heldout", default=None)
    p_scan.add_argument("--allow-training", action="store_true")
    p_scan.add_argument("--allow-redistribution", action="store_true")
    p_scan.set_defaults(func=cmd_corpus_scan)

    p_build = sub.add_parser("corpus-build", help="materialize reviewed sources for training")
    p_build.add_argument("--manifest", required=True)
    p_build.add_argument("--output", required=True)
    p_build.set_defaults(func=cmd_corpus_build)

    p_search = sub.add_parser("corpus-search", help="discover quarantined GitHub repository candidates")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--manifest", required=True)
    p_search.set_defaults(func=cmd_corpus_search)

    p_imgtrain = sub.add_parser("image-train", help="train the image model from random initialization")
    source = p_imgtrain.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--manifest", default=None)
    p_imgtrain.add_argument("--config", default="configs/image-gen-small.json")
    p_imgtrain.add_argument("--out-dir", default="out/image-gen")
    p_imgtrain.add_argument("--count", type=int, default=240)
    p_imgtrain.add_argument("--val-count", type=int, default=60)
    p_imgtrain.set_defaults(func=cmd_image_train)

    p_img = sub.add_parser("image-generate", help="generate or image-to-image edit using HELIX diffusion")
    p_img.add_argument("--config", default="artifacts/image-gen/config.json")
    p_img.add_argument("--checkpoint", default="artifacts/image-gen/ckpt-final.pt")
    p_img.add_argument("--prompt", required=True)
    p_img.add_argument("--output", required=True)
    p_img.add_argument("--seed", type=int, default=1337)
    p_img.add_argument("--init-image", default=None)
    p_img.add_argument("--strength", type=float, default=.6)
    p_img.set_defaults(func=cmd_image_generate)

    p_scene = sub.add_parser("scene-render", help="render a controlled physics-informed scene")
    p_scene.add_argument("--prompt", required=True)
    p_scene.add_argument("--output", required=True)
    p_scene.add_argument("--size", type=int, default=128)
    p_scene.set_defaults(func=cmd_scene)

    p_vision = sub.add_parser("vision-train", help="train synthetic visual attribute encoder")
    p_vision.add_argument("--out-dir", default="out/vision")
    p_vision.add_argument("--count", type=int, default=240)
    p_vision.add_argument("--val-count", type=int, default=90)
    p_vision.add_argument("--steps", type=int, default=300)
    p_vision.add_argument("--batch-size", type=int, default=24)
    p_vision.add_argument("--seed", type=int, default=1337)
    p_vision.set_defaults(func=cmd_vision_train)

    p_analyze = sub.add_parser("image-analyze", help="inspect pixels and optional synthetic-domain predictions")
    p_analyze.add_argument("--image", required=True)
    p_analyze.add_argument("--checkpoint", default="artifacts/vision/ckpt-final.pt")
    p_analyze.add_argument("--caption", action="store_true")
    p_analyze.add_argument("--caption-dir", default="artifacts/caption")
    p_analyze.set_defaults(func=cmd_image_analyze)

    p_caption = sub.add_parser("caption-train", help="train the synthetic-domain ViT-to-HELIX captioner")
    p_caption.add_argument("--steps", type=int, default=800)
    p_caption.add_argument("--batch-size", type=int, default=16)
    p_caption.add_argument("--out-dir", default="out/caption")
    p_caption.add_argument("--base-checkpoint", default="artifacts/llm/ckpt-best.pt")
    p_caption.set_defaults(func=cmd_caption_train)

    p_visual_eval = sub.add_parser("visual-eval", help="compare learned-only and physics-guided synthesis")
    p_visual_eval.add_argument("--config", default="artifacts/image-gen/config.json")
    p_visual_eval.add_argument("--image-checkpoint", default="artifacts/image-gen/ckpt-final.pt")
    p_visual_eval.add_argument("--vision-checkpoint", default="artifacts/vision/ckpt-final.pt")
    p_visual_eval.add_argument("--out-path", default="out/physics-comparison.json")
    p_visual_eval.add_argument("--seed", type=int, default=2026)
    p_visual_eval.set_defaults(func=cmd_visual_eval)

    p_math = sub.add_parser("research", help="bounded SymPy verification")
    p_math.add_argument("--operation", choices=["differentiate", "integrate", "simplify", "solve", "limit"], required=True)
    p_math.add_argument("--expression", required=True)
    p_math.add_argument("--variable", default="x")
    p_math.add_argument("--point", default="0")
    p_math.set_defaults(func=cmd_research)

    p_code = sub.add_parser("code-run", help="execute an approved file in the Linux isolation sandbox")
    p_code.add_argument("--file", required=True)
    p_code.add_argument("--approve", action="store_true")
    p_code.add_argument("--timeout", type=float, default=15)
    p_code.set_defaults(func=cmd_code_run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
