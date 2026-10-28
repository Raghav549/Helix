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
import sys

from helix.train import train as train_fn


def cmd_train(args: argparse.Namespace) -> int:
    train_fn(args.config)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
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

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
