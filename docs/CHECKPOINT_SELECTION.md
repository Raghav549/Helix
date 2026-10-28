# Chat checkpoint selection

Every training run evaluates both the base checkpoint and the instruction-tuned checkpoint using the same held-out token windows and the same small diagnostic task suite. The selector chooses the SFT checkpoint only when it passes more diagnostic tasks, or when task scores tie and its held-out language-model loss is lower. Otherwise, it keeps the base checkpoint. The selected stage, both score summaries, the reason, and the exact task list are recorded in `checkpoint-selection.json` and embedded in the model bundle.

This prevents a reduction in SFT-pair loss from being treated as proof of improved general capability. It is still only a small diagnostic suite: neither a tie nor a pass-rate improvement proves broad intelligence, factual coverage, or real-world reliability. Both base and SFT reports remain attached to each release so the decision can be audited.
