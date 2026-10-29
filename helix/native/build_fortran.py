#!/usr/bin/env python3
"""Compile the HELIX Fortran kernels with f2py, if a Fortran compiler exists.

The compiled module is optional: helix/numerics.py falls back to an equivalent
pure-Python implementation and reports which backend produced a result. Nothing
in the deployment pipeline requires Fortran, so Render and CI stay compatible.

Usage:
    python helix/native/build_fortran.py [--output helix/native]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def compiler_available() -> bool:
    return shutil.which('gfortran') is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='helix/native',
                        help='directory where the compiled extension is written')
    parser.add_argument('--require-compiler', action='store_true',
                        help='fail instead of reporting "skipped" when gfortran is absent')
    args = parser.parse_args()
    if not compiler_available():
        status = {'ok': False, 'skipped': True,
                  'reason': 'gfortran not found; helix.numerics uses the pure-Python fallback'}
        print(json.dumps(status, indent=2))
        return 1 if args.require_compiler else 0
    source = Path(__file__).with_name('kernels.f90')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'numpy.f2py', '-c', '-m', 'helix_kernels',
               str(source), '--f90flags=-O2', '--quiet']
    completed = subprocess.run(command, cwd=str(output), capture_output=True, text=True)
    report = {'ok': completed.returncode == 0, 'skipped': False,
              'python': sys.version.split()[0],
              'platlib': sysconfig.get_paths().get('platlib'),
              'stdout_tail': completed.stdout[-2000:], 'stderr_tail': completed.stderr[-2000:]}
    print(json.dumps(report, indent=2))
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
