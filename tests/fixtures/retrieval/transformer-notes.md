# Transformer notes

## Attention masking

A causal attention mask prevents position t from attending to positions greater
than t. Without the mask, a language model can read the token it is supposed to
predict, which leaks the training target into the input.

The mask is applied before the softmax, using negative infinity on masked
positions so their probability becomes zero.

## Rotary positions

Rotary positional embeddings encode absolute position by rotating query and key
vectors in pairs of dimensions. Relative offsets then appear as a phase
difference, which the attention score recovers directly.
