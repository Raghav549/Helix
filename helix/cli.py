"""HELIX command-line interface.

Subcommands:
  train    - train from random initialization (CPU baseline)
  generate - generate text from a prompt
  chat     - interactive chat
  serve    - tiny local HTTP generation server (stdlib)
  eval     - report validation perplexity from a checkpoint
"""

from __future__ import annotations

import argparse
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
    val_ids = load_ids(cfg.data.val_file)
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_iters):
            X, Y = get_batch(
                "val",
                np.load(cfg.data.train_file),
                val_ids,
                cfg.train.batch_size,
                cfg.train.block_size,
                cfg.train.device,
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
        report = run_eval(args.config, args.checkpoint, args.out_path, tasks=tasks,
                          device=cfg.train.device)
        print(f"[eval] report: {report.summary()['passed']}/{report.summary()['total']} "
              f"tasks passed -> {args.out_path}")
    except Exception as exc:  # pragma: no cover - eval harness is best-effort
        print(f"[eval] task-suite skipped: {exc}")
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
              block_sizes=[int(s) for s in args.block_sizes.split(",") if s.strip()])
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
                  require_approval=not args.no_approval)
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
    argv += ["--host", args.host, "--port", str(args.port)]
    _sys.argv = ["helix"] + argv  # drop consumed "web" token
    return web_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helix", description="HELIX CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train from random initialization")
    p_train.add_argument("--config", required=True, help="config JSON path")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="generate text from a prompt")
    p_gen.add_argument("--config", required=True)
    p_gen.add_argument("--checkpoint", required=True)
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
    p_chat.add_argument("--config", required=True)
    p_chat.add_argument("--checkpoint", required=True)
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("eval", help="validation loss / perplexity")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--eval-iters", type=int, default=50)
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
    p_sft.set_defaults(func=cmd_sft)

    p_agent = sub.add_parser("agent", help="run a modular agent with sandboxed tools")
    p_agent.add_argument("--config", required=True)
    p_agent.add_argument("--checkpoint", required=True)
    p_agent.add_argument("--goal", required=True, help="task goal for the agent")
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.add_argument("--max-time", type=float, default=60.0)
    p_agent.add_argument("--no-approval", action="store_true",
                         help="auto-approve consequential tools (demo only)")
    p_agent.add_argument("--audit-log", default="out/agent-audit.json",
                         help="path to write the audit log")
    p_agent.set_defaults(func=cmd_agent)

    p_web = sub.add_parser("web", help="local integrated web UI (chat, images, code, research)")
    p_web.add_argument("--config", default="configs/helix-tiny.json")
    p_web.add_argument("--checkpoint", default="out/tiny-run/ckpt-best.pt")
    p_web.add_argument("--image-gen-config", default=None)
    p_web.add_argument("--image-gen-checkpoint", default=None)
    p_web.add_argument("--vision-config", default=None)
    p_web.add_argument("--vision-checkpoint", default=None)
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8766)
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
