# HELIX Model Card (baseline v0.1)

## Model

- **Architecture:** independently implemented decoder-only transformer
- **Training:** from random initialization (no pretrained weights)
- **Layers / heads / embd:** 2 / 4 / 128 (tiny baseline)
- **Position:** RoPE
- **Norm / FFN:** RMSNorm / SwiGLU
- **Attention:** causal, grouped-query (n_kv_head=2)
- **Tokenizer:** byte-level BPE (from scratch)
- **Framework:** PyTorch CPU

## Intended use

Research prototype and reproducible baseline. NOT a production assistant.
Not evaluated on safety, fairness, or downstream benchmarks.

## Factors

- Trained on a small internal seed corpus only (data/seed_corpus.txt).
- CPU-only; modest capacity; short context (128 tokens).

## Metrics

Validation loss / perplexity reported by `helix eval`.

## Limitations

- Tiny model; not knowledgeable; will hallucinate.
- No safety filters, no instruction tuning, no preference alignment.
- Outputs should not be used for medical, legal, or safety-critical decisions.

## License

Code: MIT. Data/model/dependencies carry their own terms; see docs/ and
license inventories before redistribution.
