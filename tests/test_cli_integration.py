"""Subprocess CLI checks exercise the installed entry point, not just helpers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, 'HELIX_THREADS': '2', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}


def cli(*args, timeout=30):
    return subprocess.run([sys.executable, '-m', 'helix', *map(str, args)], cwd=ROOT,
                          env=ENV, capture_output=True, text=True, timeout=timeout)


def test_cli_help_math_and_execution_approval(tmp_path):
    result = cli('--help')
    assert result.returncode == 0 and 'caption-train' in result.stdout and 'web' in result.stdout
    result = cli('research', '--operation', 'differentiate', '--expression', 'x**3 + 2*x')
    assert result.returncode == 0 and json.loads(result.stdout)['result'] == '3*x**2 + 2'
    file = tmp_path / 'code.py'; file.write_text('result = 42\n')
    assert cli('code-run', '--file', file).returncode == 2
    result = cli('code-run', '--file', file, '--approve')
    data = json.loads(result.stdout)
    if not data['ok'] and 'isolation unavailable' in (data['error'] or ''):
        pytest.skip('host sandbox unavailable; CLI failed closed')
    assert result.returncode == 0 and '42' in data['output']


def test_cli_controlled_scene_and_visual_analysis(tmp_path):
    out = tmp_path/'scene.png'
    result = cli('scene-render', '--prompt', 'a blue cube on the right', '--output', out)
    assert result.returncode == 0 and out.is_file()
    result = cli('image-analyze', '--image', out, '--caption')
    if not (ROOT/'artifacts/caption/ckpt-best.pt').exists():
        pytest.skip('trained artifacts absent')
    assert result.returncode == 0, result.stderr
    assert 'synthetic_domain_caption' in json.loads(result.stdout)


def test_cli_eval_writes_seeded_validation_report(tmp_path):
    if not (ROOT/'artifacts/llm/ckpt-best.pt').exists(): pytest.skip('trained artifacts absent')
    out = tmp_path/'eval.json'
    result = cli('eval', '--config', 'configs/helix-research.json', '--checkpoint', 'artifacts/llm/ckpt-best.pt',
                 '--eval-iters', '2', '--seed', '123', '--eval-tasks', 'calibration_true_vs_false', '--out-path', out)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report['validation']['batches'] == 2 and report['validation']['seed'] == 123
    assert report['total'] == 1
