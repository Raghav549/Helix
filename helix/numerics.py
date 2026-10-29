"""Deterministic numerical helpers with an optional Fortran backend.

What this module is: least-squares trend fitting with a prediction interval, an
EWMA smoother, autoregressive and multiple-regression forecasting, polynomial
trend fitting, correlation and error metrics. What it is not: a forecasting
model that understands a time series, a learned predictor, or evidence about any
real system. Every result carries the assumptions, the fit window, the hold-out
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
AR_MAX_LAG = 12
MLR_MAX_FEATURES = 12
POLY_MAX_DEGREE = 12


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


# ---------------------------------------------------------------------------
# Ordinary least squares (univariate)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exponentially weighted moving average
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise NumericInputError('actual and predicted must be equally long and non-empty')
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise NumericInputError('actual and predicted must be equally long and non-empty')
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


# ---------------------------------------------------------------------------
# Autoregressive (AR(p)) forecasting: predict from the series' own past.
# ---------------------------------------------------------------------------

def _ols_fit_matrix(X: List[List[float]], y: List[float]) -> tuple:
    """Pure-python OLS for [1, features] design; returns (intercept, coefs)."""
    n = len(X)
    ncol = len(X[0])
    # normal equations
    xtx = [[0.0] * ncol for _ in range(ncol)]
    xty = [0.0] * ncol
    for i in range(n):
        for j in range(ncol):
            xty[j] += X[i][j] * y[i]
            for m in range(ncol):
                xtx[j][m] += X[i][j] * X[i][m]
    # Gaussian elimination with partial pivoting
    import copy
    a = [row[:] for row in xtx]
    bb = xty[:]
    n_ = ncol
    for col in range(n_):
        piv = max(range(col, n_), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-14:
            return None
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            bb[col], bb[piv] = bb[piv], bb[col]
        pivval = a[col][col]
        for r in range(col + 1, n_):
            factor = a[r][col] / pivval
            a[r][col] = factor
            for c in range(col + 1, n_):
                a[r][c] -= factor * a[col][c]
            bb[r] -= factor * bb[col]
    if abs(a[n_ - 1][n_ - 1]) < 1e-14:
        return None
    x = [0.0] * n_
    x[n_ - 1] = bb[n_ - 1] / a[n_ - 1][n_ - 1]
    for i in range(n_ - 2, -1, -1):
        s = bb[i]
        for j in range(i + 1, n_):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x[0], x[1:]


def ar_fit(series: Sequence[float], p: int) -> dict:
    """Fit x_t = intercept + phi_1 x_{t-1} + ... + phi_p x_{t-p} by OLS.

    Returns coefficients usable by ``ar_predict`` plus hold-out diagnostics.
    """
    series = _clean_series(series)
    n = len(series)
    if not 1 <= int(p) <= AR_MAX_LAG:
        raise NumericInputError(f'p must be in 1..{AR_MAX_LAG}')
    p = int(p)
    if n <= p + 2:
        raise NumericInputError(f'needs more than p + 2 points for AR({p})')

    if _F2PY:  # pragma: no cover
        phi, intercept, sd = helix_kernels.ar_fit(
            [float(v) for v in series], n, p)
        phi = [float(v) for v in phi]
    else:
        X = [[1.0] + [float(series[p + i - j]) for j in range(1, p + 1)]
             for i in range(n - p)]
        y = [float(series[p + i]) for i in range(n - p)]
        solved = _ols_fit_matrix(X, y)
        if solved is None:
            raise NumericInputError('AR design matrix is singular; reduce p')
        intercept, phi = solved
        # residual SD
        sse = 0.0
        for i in range(len(y)):
            pred = intercept + sum(c * X[i][j + 1] for j, c in enumerate(phi))
            sse += (y[i] - pred) ** 2
        m = len(y) - (p + 1)
        sd = math.sqrt(sse / m) if m > 0 else 0.0

    # one-step-ahead fitted values on the training range for a sanity score
    fitted = []
    for t in range(p, n):
        pred = intercept + sum(phi[j] * series[t - 1 - j] for j in range(p))
        fitted.append(pred)
    insample_rmse = rmse(series[p:], fitted) if fitted else 0.0

    return {
        'backend': BACKEND,
        'tool': 'autoregressive fit (AR(p))',
        'p': p,
        'points_used': n,
        'intercept': round(intercept, 8),
        'phi': [round(v, 8) for v in phi],
        'residual_sd': round(sd, 8),
        'insample_rmse': round(insample_rmse, 8),
        'assumptions': [
            'The series is modelled as a linear function of its own p lags.',
            'No exogenous regressors, seasonality or regime changes are modelled.',
            'Coefficients are fitted by OLS on the full series.',
        ],
        'limitations': ('An AR(p) extrapolates the series from its own history; '
                        'it is not evidence about causes. Use with the reported '
                        'insample_rmse and residual_sd.'),
    }


def ar_predict(series: Sequence[float], p: int, steps: int) -> dict:
    """Iteratively forecast ``steps`` future values from the fitted AR(p)."""
    series = _clean_series(series)
    n = len(series)
    if not 1 <= int(p) <= AR_MAX_LAG:
        raise NumericInputError(f'p must be in 1..{AR_MAX_LAG}')
    p = int(p)
    if not 1 <= int(steps) <= 120:
        raise NumericInputError('steps must be 1..120')
    steps = int(steps)
    if n <= p + 1:
        raise NumericInputError(f'needs more than p + 1 points for AR({p})')

    fit = ar_fit(series, p)
    intercept = fit['intercept']
    phi = fit['phi']

    if _F2PY:  # pragma: no cover
        out = helix_kernels.ar_predict(
            [float(v) for v in series], n, p, phi, intercept, steps)
        forecasts = [float(v) for v in out]
    else:
        buf = [float(v) for v in series]
        forecasts = []
        for _ in range(steps):
            nxt = intercept
            for j in range(p):
                nxt = nxt + phi[j] * buf[len(buf) - 1 - j]
            forecasts.append(nxt)
            buf.append(nxt)

    return {
        'backend': BACKEND,
        'tool': 'autoregressive forecast (AR(p))',
        'p': p,
        'points_used': n,
        'steps': steps,
        'forecasts': [round(v, 8) for v in forecasts],
        'last_observed': round(series[-1], 8),
        'note': ('Each step is conditioned on the previously forecasted value, '
                 'so uncertainty compounds with the horizon.'),
    }


# ---------------------------------------------------------------------------
# Multiple linear regression: predict from several data columns.
# ---------------------------------------------------------------------------

def mlr_fit(X: Sequence[Sequence[float]], y: Sequence[float]) -> dict:
    """Fit y = intercept + beta_1 x_1 + ... + beta_k x_k by OLS.

    X is a list of rows (one per observation); y is the target.
    """
    rows = []
    for row in X:
        row_vals = [float(v) for v in row]
        if any(not math.isfinite(v) for v in row_vals):
            raise NumericInputError('feature values must be finite numbers')
        rows.append(row_vals)
    targets = _clean_series(y)
    if len(rows) != len(targets) or len(rows) < 2:
        raise NumericInputError('X and y must be equally long with at least 2 rows')
    n = len(rows)
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        raise NumericInputError('all X rows must have the same number of features')
    if not 1 <= k <= MLR_MAX_FEATURES:
        raise NumericInputError(f'number of features must be 1..{MLR_MAX_FEATURES}')
    if n < k + 2:
        raise NumericInputError(f'needs at least k + 2 = {k + 2} rows for {k} features')

    if _F2PY:  # pragma: no cover
        beta, intercept, sd, r2 = helix_kernels.mlr_fit(
            rows, targets, n, k)
        beta = [float(v) for v in beta]
    else:
        solved = _ols_fit_matrix([[1.0] + r for r in rows], targets)
        if solved is None:
            raise NumericInputError('feature matrix is singular; drop collinear features')
        intercept, beta = solved
        sse = 0.0
        sst = 0.0
        mean_y = sum(targets) / n
        for i in range(n):
            pred = intercept + sum(c * r for c, r in zip(beta, rows[i]))
            sse += (targets[i] - pred) ** 2
            sst += (targets[i] - mean_y) ** 2
        m = n - (k + 1)
        sd = math.sqrt(sse / m) if m > 0 else 0.0
        r2 = 1.0 - sse / sst if sst > 0 else 0.0

    return {
        'backend': BACKEND,
        'tool': 'multiple linear regression (OLS)',
        'points_used': n,
        'features': k,
        'intercept': round(intercept, 8),
        'beta': [round(v, 8) for v in beta],
        'residual_sd': round(sd, 8),
        'r2': round(r2, 8),
        'assumptions': [
            'Linear relationship between features and target with independent noise.',
            'No regularisation or feature selection is applied.',
        ],
        'limitations': ('Coefficients describe association in this data, not causation. '
                        'Extrapolation beyond the observed feature range is unreliable.'),
    }


def mlr_predict(x_row: Sequence[float], fit: dict) -> dict:
    """Predict the target for one feature row using a fitted MLR model."""
    row = [float(v) for v in x_row]
    if len(row) != fit['features']:
        raise NumericInputError(f'row must have {fit["features"]} features')
    yhat = fit['intercept'] + sum(b * v for b, v in zip(fit['beta'], row))
    return {'yhat': round(yhat, 8), 'features': len(row)}


# ---------------------------------------------------------------------------
# Polynomial trend fit
# ---------------------------------------------------------------------------

def poly_fit(series: Sequence[float], degree: int) -> dict:
    """Fit y = c_0 + c_1 t + ... + c_d t^d by least squares over t = 1..n."""
    series = _clean_series(series)
    n = len(series)
    if not 1 <= int(degree) <= POLY_MAX_DEGREE:
        raise NumericInputError(f'degree must be 1..{POLY_MAX_DEGREE}')
    degree = int(degree)
    if n < degree + 3:
        raise NumericInputError(f'needs at least degree + 3 = {degree + 3} points')

    if _F2PY:  # pragma: no cover
        coef, sd = helix_kernels.poly_fit(
            [float(v) for v in series], n, degree)
        coef = [float(v) for v in coef]
    else:
        X = [[float(i) ** j for j in range(degree + 1)] for i in range(1, n + 1)]
        solved = _ols_fit_matrix(X, series)
        if solved is None:
            raise NumericInputError('polynomial design matrix is singular; reduce degree')
        intercept, beta = solved
        coef = [intercept] + beta
        sse = 0.0
        for i in range(n):
            pred = sum(c * (float(i + 1) ** j) for j, c in enumerate(coef))
            sse += (series[i] - pred) ** 2
        m = n - (degree + 1)
        sd = math.sqrt(sse / m) if m > 0 else 0.0

    return {
        'backend': BACKEND,
        'tool': 'polynomial trend fit (least squares)',
        'points_used': n,
        'degree': degree,
        'coef': [round(v, 8) for v in coef],
        'residual_sd': round(sd, 8),
        'assumptions': ['The index (1, 2, ..., n) is the only regressor.'],
        'limitations': ('High-degree polynomials oscillate and extrapolate poorly. '
                        'Prefer degree 1 or 2 unless the data clearly warrant more.'),
    }


def poly_predict(degree: int, coef: Sequence[float], t: float) -> dict:
    """Evaluate the fitted polynomial at index t."""
    yhat = sum(c * (float(t) ** j) for j, c in enumerate(coef))
    return {'t': float(t), 'yhat': round(yhat, 8)}


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def correlation(X: Sequence[Sequence[float]]) -> dict:
    """Pearson correlation matrix of k series (rows), each of length n."""
    rows = [[float(v) for v in row] for row in X]
    if not rows or any(not row for row in rows):
        raise NumericInputError('X must be non-empty rows of numbers')
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise NumericInputError('all rows must have the same length')
    if n < 3:
        raise NumericInputError('each row needs at least 3 points')
    k = len(rows)
    if any(not math.isfinite(v) for row in rows for v in row):
        raise NumericInputError('all values must be finite')

    means = [sum(r) / n for r in rows]
    sds = []
    for r, mean in zip(rows, means):
        var = sum((v - mean) ** 2 for v in r) / max(n - 1, 1)
        sds.append(math.sqrt(var))
    eps = 1e-12
    corr = [[1.0] * k for _ in range(k)]
    for j in range(k):
        for m in range(j + 1, k):
            cov = sum((rows[j][i] - means[j]) * (rows[m][i] - means[m])
                      for i in range(n)) / max(n - 1, 1)
            if sds[j] > eps and sds[m] > eps:
                c = cov / (sds[j] * sds[m])
            else:
                c = 0.0
            c = max(-1.0, min(1.0, c))
            corr[j][m] = c
            corr[m][j] = c
    return {
        'backend': BACKEND,
        'tool': 'Pearson correlation matrix',
        'series': k,
        'points': n,
        'corr': [[round(v, 6) for v in row] for row in corr],
    }


# ---------------------------------------------------------------------------
# High-level linear forecast (kept for the existing orchestrator tool)
# ---------------------------------------------------------------------------

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
