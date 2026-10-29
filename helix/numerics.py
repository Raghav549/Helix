"""Deterministic numerical helpers with an optional Fortran backend.

What this module is: least-squares trend fitting with a prediction interval, an
EWMA smoother and error metrics. What it is not: a forecasting model that
understands a time series, a learned predictor, or evidence about any real
system. Every result carries the assumptions, the fit window, the hold-out
score and the backend that produced it, so a caller can judge it.

The Fortran kernels in ``helix/native/kernels.f90`` are used only when compiled
with f2py; otherwise the identical pure-Python implementation runs.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - depends on an optional compiled extension
    import helix_kernels  # type: ignore
    _F2PY = True
except Exception:  # pragma: no cover - the normal path without a compiler
    helix_kernels = None  # type: ignore
    _F2PY = False

BACKEND = 'fortran' if _F2PY else 'python'
MIN_POINTS = 3
MAX_POINTS = 5000


def backend_name() -> str:
    return BACKEND


class NumericInputError(ValueError):
    """Raised for malformed or out-of-range numeric input."""


def _clean_series(values: Iterable[float]) -> List[float]:
    series: List[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NumericInputError('series values must be finite numbers')
        number = float(value)
        if not math.isfinite(number):
            raise NumericInputError('series values must be finite numbers')
        series.append(number)
    if len(series) < MIN_POINTS:
        raise NumericInputError(f'at least {MIN_POINTS} numeric points are required')
    if len(series) > MAX_POINTS:
        raise NumericInputError(f'at most {MAX_POINTS} points are supported')
    return series


def ols_fit(x: Sequence[float], y: Sequence[float]) -> dict:
    """Fit y = intercept + slope * x, returning slope, intercept, sd and R²."""
    if len(x) != len(y) or len(x) < 2:
        raise NumericInputError('x and y must be equally long and contain at least 2 points')
    if _F2PY:  # pragma: no cover - exercised only with a compiled kernel
        slope, intercept, sd, r2 = helix_kernels.ols_fit(
            [float(v) for v in x], [float(v) for v in y])
        return {'slope': float(slope), 'intercept': float(intercept),
                'residual_sd': float(sd), 'r2': float(r2), 'n': len(x)}
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    syy = sum((yi - mean_y) ** 2 for yi in y)
    if sxx <= 0:
        return {'slope': 0.0, 'intercept': mean_y, 'residual_sd': 0.0, 'r2': 0.0, 'n': n}
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    residual_sd = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    r2 = 1.0 - sse / syy if syy > 0 else 0.0
    return {'slope': slope, 'intercept': intercept, 'residual_sd': residual_sd,
            'r2': r2, 'n': n}


def ols_predict(x0: float, x: Sequence[float], fit: dict, z: float) -> dict:
    """Predict at x0 with a *prediction* interval (not a confidence interval)."""
    mean_x = sum(x) / len(x)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    yhat = fit['intercept'] + fit['slope'] * x0
    if _F2PY:  # pragma: no cover
        yhat, lower, upper = helix_kernels.ols_predict(x0, len(x), mean_x, sxx, fit['slope'],
                                                      fit['intercept'], fit['residual_sd'], z)
        return {'x': x0, 'yhat': float(yhat), 'lower': float(lower), 'upper': float(upper)}
    margin = 0.0
    if sxx > 0 and len(x) > 2:
        margin = z * fit['residual_sd'] * math.sqrt(1.0 + 1.0 / len(x) + ((x0 - mean_x) ** 2) / sxx)
    return {'x': x0, 'yhat': yhat, 'lower': yhat - margin, 'upper': yhat + margin}


def ewma(values: Sequence[float], alpha: float) -> List[float]:
    """Exponentially weighted moving average with 0 < alpha <= 1."""
    if not 0 < alpha <= 1:
        raise NumericInputError('alpha must be in (0, 1]')
    series = [float(value) for value in values]
    if not series:
        return []
    if _F2PY:  # pragma: no cover
        return [float(value) for value in helix_kernels.ewma(series, alpha)]
    out = [series[0]]
    for value in series[1:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise NumericInputError('actual and predicted must be equally long and non-empty')
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise NumericInputError('actual and predicted must be equally long and non-empty')
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def linear_forecast(values: Iterable[float], horizon: int = 3, interval: float = 0.90,
                    holdout_fraction: float = 0.2) -> dict:
    """Fit a linear trend, score it on a hold-out tail, then forecast.

    The reported error metrics come from the hold-out tail only. The forecast is
    refitted on the full series, and the interval is a prediction interval for a
    single future observation under the linear-trend assumption.
    """
    series = _clean_series(values)
    if not 1 <= int(horizon) <= 60:
        raise NumericInputError('horizon must be 1..60 steps')
    if not 0.50 <= float(interval) < 1.0:
        raise NumericInputError('interval must be in [0.50, 1.0)')
    if not 0.0 < float(holdout_fraction) < 0.5:
        raise NumericInputError('holdout_fraction must be in (0, 0.5)')

    horizon = int(horizon)
    n = len(series)
    holdout = max(2, int(round(n * holdout_fraction))) if n >= 8 else 2
    holdout = min(holdout, n - 2)
    train = series[: n - holdout]
    test = series[n - holdout:]
    x_train = [float(i) for i in range(len(train))]
    fit_holdout = ols_fit(x_train, train)
    predicted = [fit_holdout['intercept'] + fit_holdout['slope'] * (len(train) + j)
                 for j in range(len(test))]
    naive = [train[-1]] * len(test)

    x_all = [float(i) for i in range(n)]
    fit = ols_fit(x_all, series)
    z = NormalDist().inv_cdf(0.5 + interval / 2)
    predictions = [ols_predict(float(n + step), x_all, fit, z) for step in range(horizon)]

    return {
        'backend': BACKEND,
        'tool': 'least-squares trend with prediction interval',
        'points_used': n,
        'fit_window': {'train': len(train), 'holdout': holdout},
        'fit': {key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in fit.items()},
        'holdout': {
            'mae': round(mae(test, predicted), 6),
            'rmse': round(rmse(test, predicted), 6),
            'baseline_mae_last_value': round(mae(test, naive), 6),
            'beats_baseline': mae(test, predicted) < mae(test, naive),
        },
        'predictions': [{key: round(value, 6) for key, value in point.items()}
                        for point in predictions],
        'interval': interval,
        'assumptions': [
            'The series is modelled as a straight-line trend plus independent noise.',
            'The index (0, 1, 2, ...) is the only regressor; seasonality is not modelled.',
            'The interval is a prediction interval for one future value, not a confidence band.',
            'Errors are measured on the chronological hold-out tail only.',
        ],
        'limitations': ('A fitted trend is not evidence about causes or about a system that '
                        'changes its behaviour. Report the horizon and baseline with any use.'),
    }
