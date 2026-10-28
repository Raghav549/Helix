# HELIX deployment notes

## Render CPU demo (not a training host)

The Render blueprint intentionally runs a single Gunicorn worker with two
threads and CPU-only PyTorch. It does not launch training at web-service startup.
Render Free services sleep when inactive and have no persistent disk, so local
checkpoints, memory, and generated images are ephemeral. Do not store sensitive
notes or assume durable history in this demo.

## Required checks before public sharing

- The default app has no authentication. Treat any public deployment as a public
  demo only after access controls/rate limits have been implemented. Do not enter
  secrets into it or use code execution with untrusted users.
- The bundled default checkpoint paths must exist in the runtime image. If the
  required model artifacts are not shipped in the repo or fetched from
  separately hosted, hash-verified object storage, the service should report
  model unavailable rather than pretend the model is loaded.
- Render runs `render.yaml` from the branch where it exists; merge the changes
  and then create/apply the Blueprint in the Dashboard.
- Set `HELIX_DATA_ROOT` to a persistent external volume for any deployment that
  needs durable checkpoints or data. Render Free web services do not provide one.
- Full-scale training should run on a dedicated compute environment, not the
  free web process. Training needs to be a separately managed resumable job with
  explicit time, disk, and memory budgets.

## Reproducing locally

Install the official CPU PyTorch wheel, install `requirements.txt`, then run:

```sh
python -m pytest -q
python -m helix doctor
python -m helix web --host 127.0.0.1 --port 8766
```
