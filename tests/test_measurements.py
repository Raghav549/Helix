"""Guards for the committed measurement evidence.

The numbers in docs/MEASUREMENTS.md come from docs/measurements/*.json. These
tests keep the two consistent so a stale or hand-edited report cannot ship.
"""
import json
from pathlib import Path

import pytest

from helix.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / 'docs' / 'measurements'


def _load(name):
    path = MEASUREMENTS / name
    assert path.is_file(), f'missing measurement artifact: {name}'
    return json.loads(path.read_text(encoding='utf-8'))


def test_reproduce_configs_load_and_match_the_documented_geometry():
    for name, iters in (('reproduce-cpu.json', 200), ('reproduce-cpu-sft.json', 60)):
        config = load_config(ROOT / 'configs' / name)
        assert config.model.vocab_size == 512
        assert config.model.block_size == 128
        assert config.model.n_layer == 3
        assert config.model.n_embd == 128
        assert config.train.max_iters == iters
        assert config.train.seed == 1337


def test_pretraining_report_carries_provenance_and_environment():
    metrics = _load('pretrain-metrics.json')
    assert metrics['parameters'] == 803_712
    assert metrics['skipped_steps'] == 0
    assert metrics['best_validation_loss'] == pytest.approx(4.78095, abs=1e-4)
    provenance = metrics['provenance']
    assert len(provenance['git_commit']) == 40
    assert len(provenance['corpus']['sha256']) == 64
    assert metrics['environment']['device'] == 'cpu'
    assert '<run-root>' in provenance['corpus']['path'] or not provenance['corpus']['path'].startswith('/')


def test_sft_report_split_adds_up_and_is_hashed():
    sft = _load('sft-metrics.json')
    assert sft['train_pairs'] + sft['heldout_pairs'] == sft['n_pairs']
    assert sft['heldout_pairs'] > 0, 'instruction tuning must keep a held-out split'
    assert len(sft['pairs_sha256']) == 64
    assert sft['seed'] == 1337


def test_evaluation_summary_matches_its_own_task_results():
    report = _load('eval.json')
    passed = sum(1 for task in report['results'] if task['pass_'])
    assert report['passed'] == passed
    assert report['total'] == len(report['results'])
    assert report['passed'] < report['total'], 'the documented model must not look better than it is'
    calibration = next(task for task in report['results'] if task['task'] == 'calibration_true_vs_false')
    assert calibration['pass_'] is True
    for task in report['results']:
        if task['task'] == 'calibration_true_vs_false':
            continue
        assert task['value'] < task['threshold'], f'{task["task"]} unexpectedly passed; update the docs honestly'


def test_benchmark_report_contains_the_measured_kv_cache_comparison():
    report = _load('bench-report.json')
    results = {item['name']: item['value'] for item in report['results']}
    assert results['generate_kvcache_speedup_vs_full'] > 1.0
    assert results['generate_kvcache_output_matches_full'] == 1.0
    assert report['hardware']['has_cuda'] is False


def test_measurements_document_references_the_committed_reports():
    text = (ROOT / 'docs' / 'MEASUREMENTS.md').read_text(encoding='utf-8')
    for name in ('pretrain-metrics.json', 'sft-metrics.json', 'eval.json', 'bench-report.json'):
        assert (MEASUREMENTS / name).is_file()
    assert 'docs/measurements/' in text
    assert '803,712' in text
    assert 'not a general' in text.lower() or 'do not show fluent' in text.lower()
