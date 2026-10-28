"""Provenance, tokenizer split, and training-resume correctness."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from helix.config import DataConfig, HelixConfig, ModelConfig, TrainConfig, save_config
from helix.corpus import RIGHTS_TRAIN, build_training_corpus, scan_directory
from helix.data import build_from_corpus
from helix.train import train


def test_rights_quarantine_duplicates_and_hash_mutation(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    text = 'A useful independent document with enough original text for the data pipeline.'
    (source / 'a.txt').write_text(text)
    (source / 'b.txt').write_text(text)
    unknown = scan_directory(str(source))
    assert len(unknown.trainable()) == 0 and len(unknown.quarantined()) == 2
    reviewed = scan_directory(str(source), rights=[RIGHTS_TRAIN], default_license='MIT', license_evidence='local LICENSE')
    assert len(reviewed.trainable()) == 1
    assert any(s.quarantine_reason == 'exact duplicate' for s in reviewed.quarantined())
    dest = tmp_path / 'corpus.txt'
    build_training_corpus(reviewed, str(dest))
    (source / 'a.txt').write_text(text + ' changed')
    with pytest.raises(ValueError, match='changed since review'):
        build_training_corpus(reviewed, str(dest))


def test_bpe_is_trained_only_on_training_documents(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('aaaa aaa aaaa\n\nbbbb bbb bbbb\n\ncccc ccc cccc\n\nZZZZ ZZZZ ZZZZ\n')
    tok, train_ids, val_ids = build_from_corpus(str(corpus), str(tmp_path/'vocab.json'), str(tmp_path/'merges.json'),
                                                280, .75, str(tmp_path/'train.npy'), str(tmp_path/'val.npy'))
    assert all(b'ZZ' not in piece for piece in tok.vocab.values())
    eos = tok._id_for_bytes(b'<|endoftext|>')
    assert list(train_ids).count(eos) == 3 and list(val_ids).count(eos) == 1
    assert all(eos not in (a, b) for a, b, _ in tok.merges)
    metadata = json.loads((tmp_path/'train.npy.meta.json').read_text())
    assert metadata['tokenizer_trained_on'] == 'train only' and len(metadata['file_sha256']) == 4


def test_resume_matches_uninterrupted_and_rejects_stale_data(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(('alpha beta gamma delta epsilon\n\n' * 8) + 'held out example text\n')
    data = DataConfig(train_file=str(tmp_path/'train.npy'), val_file=str(tmp_path/'val.npy'),
                      vocab_file=str(tmp_path/'vocab.json'), merges_file=str(tmp_path/'merges.json'),
                      corpus_file=str(corpus), train_frac=.8)
    model = ModelConfig(vocab_size=270, block_size=16, n_layer=1, n_head=2, n_embd=32, n_kv_head=1, dropout=.1)
    base = TrainConfig(batch_size=2, block_size=16, max_iters=8, eval_interval=2, eval_iters=2,
                       learning_rate=.001, warmup_iters=1, lr_decay_iters=8, log_interval=2,
                       save_interval=0, seed=123, out_dir=str(tmp_path/'full'))
    def run(train_cfg, name):
        path = tmp_path / (name + '.json'); save_config(HelixConfig(model=model, train=train_cfg, data=data), str(path)); train(str(path))
    run(base, 'full')
    part = replace(base, max_iters=4, out_dir=str(tmp_path/'part'))
    run(part, 'part')
    resumed = replace(base, out_dir=str(tmp_path/'resumed'), resume_from=str(tmp_path/'part/ckpt-final.pt'))
    run(resumed, 'resumed')
    a = torch.load(tmp_path/'full/ckpt-final.pt', map_location='cpu', weights_only=True)
    b = torch.load(tmp_path/'resumed/ckpt-final.pt', map_location='cpu', weights_only=True)
    assert a['iter_num'] == b['iter_num'] == 8
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
    corpus.write_text(corpus.read_text() + 'altered')
    with pytest.raises(ValueError, match='corpus hash mismatch'):
        run(base, 'stale')


def test_unknown_license_cannot_be_attested_into_training(tmp_path):
    (tmp_path / 'doc.txt').write_text('A reasonably long document with words that looks suitable for training.')
    manifest = scan_directory(str(tmp_path), rights=[RIGHTS_TRAIN], default_license=' UNKNOWN ', license_evidence='somewhere')
    assert not manifest.trainable() and manifest.quarantined()
