#!/usr/bin/env python3
"""Benchmark the numeric backends and verify equivalence.

Compares:
  * helix.numerics (Fortran kernel when compiled, otherwise pure Python)
  * a NumPy vectorised reference implementation
and reports wall-clock time for a realistic workload. Run from the repo root:

    python scripts/bench_numerics.py --series 2000 --repeats 40
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from helix.numerics import BACKEND, linear_forecast, ols_fit


def numpy_reference(series: np.ndarray) -> tuple[float, float]:
    x = np.arange(series.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, series, 1)
    return float(slope), float(intercept)


def timed(function, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {'median_ms': round(statistics.median(samples), 4),
            'p95_ms': round(samples[int(len(samples) * 0.95) - 1], 4),
            'min_ms': round(samples[0], 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=40)
    args = parser.parse_args()
    rng = np.random.default_rng(1337)
    series = np.cumsum(rng.normal(0.05, 1.0, size=args.series)) + 10.0
    values = [float(value) for value in series]

    helix_fit = ols_fit([float(i) for i in range(args.series)], values)
    reference = numpy_reference(series)
    max_slope_error = abs(helix_fit['slope'] - reference[0])
    max_intercept_error = abs(helix_fit['intercept'] - reference[1])

    report = {
        'backend': BACKEND,
        'series_points': args.series,
        'repeats': args.repeats,
        'correctness': {
            'slope_abs_error_vs_numpy': max_slope_error,
            'intercept_abs_error_vs_numpy': max_intercept_error,
            'agree_within_1e-6': max_slope_error < 1e-6 and max_intercept_error < 1e-6,
        },
        'timings': {
            'helix_ols_fit': timed(lambda: ols_fit([float(i) for i in range(args.series)], values),
                                   args.repeats),
            'numpy_polyfit': timed(lambda: numpy_reference(series), args.repeats),
            'helix_linear_forecast': timed(lambda: linear_forecast(values, horizon=3),
                                          max(3, args.repeats // 4)),
        },
        'note': ('The Fortran kernel is used only when helix_kernels was compiled with f2py. '
                 'The pure-Python fallback is slower per call but has no build requirement; '
                 'both are verified against NumPy polyfit here.'),
    }
    print(json.dumps(report, indent=2))
    return 0 if report['correctness']['agree_within_1e-6'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
