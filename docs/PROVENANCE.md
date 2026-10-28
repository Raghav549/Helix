# Provenance, attribution and licenses

This inventory is evidence for the included experiments, not legal advice. A
license or safety field is a human attestation, not an automated legal ruling.
Unknown-rights sources remain quarantined. Observe the original notices and any
applicable terms when redistributing source data, corpora or model outputs.

## Project and artifacts

| Material | Origin / rights |
| --- | --- |
| HELIX-authored code, documentation, seed text, SFT pairs and synthetic renderer outputs | MIT, `LICENSE` / `data/sources/helix/LICENSE` |
| HELIX-trained `.pt` model weights | MIT, `artifacts/WEIGHTS_LICENSE.txt`; no external pretrained weights are included |
| Combined training text and tokenizer/tokens | Derived from multiple sources below; **not** a blanket MIT-only dataset. Keep third-party notices. |
| `artifacts/environment-freeze.txt` | Installed package versions for the recorded run; not a fully portable dependency lock |

Weights were trained from the mixed corpus. Generated text can reproduce
training content; the weights license does not remove third-party rights or
create a privacy guarantee. Only load checkpoints from trusted sources.

## Text/code sources used in the included base run

| Source | Pinned version | Selected files | License / notice |
| --- | --- | --- | --- |
| HELIX seed corpus | repository baseline `5137540` | `data/sources/helix/seed_corpus.txt` | MIT, `data/sources/helix/LICENSE` |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | README, `model.py`, `train.py` | MIT, copyright Andrej Karpathy/contributors, `data/sources/nanogpt/LICENSE` |
| [sympy/sympy](https://github.com/sympy/sympy) | `5b861e4c4a06d56bac4b3f1398985416ae6ec6c7` | five introductory tutorial `.rst` files | BSD-3-Clause-style and composite notices, copyright SymPy Development Team/contributors, `data/sources/sympy/LICENSE` |

`SOURCE.json` files list upstream paths; `data/manifest.json` lists each local
file, attribution, rights attestation, retrieval date, size, and SHA-256.
`scripts/prepare_corpus.py` checks hashes before materialization. The committed
corpus is 102,879 bytes; no restricted OpenStax content or pretrained weights
were included. A few benchmark strings were checked, not an exhaustive review.

Synthetic image data use only HELIX's renderer and authored templates. The
captioner starts from HELIX's own LLM checkpoint, not an outside vision or
language model. The `image-understanding` classifier, captioner, image
synthesizer and language model are distinct checkpoints as shown in the model
cards.

## Main third-party software dependencies

Package versions actually installed for the run are in
`artifacts/environment-freeze.txt`; their distributions include authoritative
license texts and may include further notices. Direct dependencies include:

| Dependency | Role | License family (check the distribution for full terms) |
| --- | --- | --- |
| PyTorch | tensor/autodiff runtime | BSD-3-Clause plus bundled notices |
| NumPy | numeric arrays | BSD-3-Clause plus bundled-library notices |
| Pillow | image processing | Pillow/PIL MIT-CMU / HPND-style notice |
| Flask / Werkzeug | local HTTP application | BSD-3-Clause |
| SymPy | independent symbolic computations | BSD-3-Clause / composite notices |
| Matplotlib | local plots | Matplotlib/PSF-style license and bundled notices |
| pypdf (optional) | text extraction from PDFs | BSD-3-Clause |
| pytest (development) | test runner | MIT |

The recorded PyPI fallback installed NVIDIA/CUDA-associated runtime packages
under their own terms despite no CUDA device being used; a CPU wheel avoids
those downloads. `libseccomp`, Linux, system Python, and util-linux `unshare` /
`mount` are system prerequisites for the isolated executor and have separate
licenses. Optional UI smoke testing uses jsdom, Puppeteer and Chromium, each
with its own license and dependency notices.
