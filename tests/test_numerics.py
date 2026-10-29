"""Fortran/native kernel equivalence and forecasting correctness tests."""
import math
import random

import pytest

from helix import numerics
from helix.numerics import (BACKEND, NumericInputError, ewma, linear_forecast, mae,
                            ols_fit, ols_predict, rmse)


def test_backend_is_reported_honestly():
    assert numerics.backend_name() in ('fortran', 'python')
    if BACKEND == 'python':
        # The compiled module is optional; the fallback must be what runs.
        assert numerics.helix_kernels is None


def test_ols_matches_closed_form_reference():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]  # exact y = 1 + 2x
    fit = ols_fit(xs, ys)
    assert fit['slope'] == pytest.approx(2.0, abs=1e-12)
    assert fit['intercept'] == pytest.approx(1.0, abs=1e-12)
    assert fit['residual_sd'] == pytest.approx(0.0, abs=1e-12)
    assert fit['r2'] == pytest.approx(1.0, abs=1e-12)
    prediction = ols_predict(5.0, xs, fit, 1.645)
    assert prediction['yhat'] == pytest.approx(11.0, abs=1e-12)
    assert prediction['lower'] == pytest.approx(11.0, abs=1e-12)


def test_ols_matches_numpy_on_noisy_data():
    rng = random.Random(7)
    xs = [float(i) for i in range(40)]
    ys = [2.5 * x + 4 + rng.gauss(0, 1.5) for x in xs]
    fit = ols_fit(xs, ys)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        pytest.skip('numpy unavailable')
    slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
    assert fit['slope'] == pytest.approx(float(slope), rel=1e-9)
    assert fit['intercept'] == pytest.approx(float(intercept), rel=1e-9)
    residuals = [y - (fit['intercept'] + fit['slope'] * x) for x, y in zip(xs, ys)]
    expected_sd = math.sqrt(sum(r * r for r in residuals) / (len(xs) - 2))
    assert fit['residual_sd'] == pytest.approx(expected_sd, rel=1e-12)


def test_prediction_interval_widens_away_from_the_mean():
    xs = [float(i) for i in range(20)]
    ys = [0.5 * x + 3 + (1.0 if x % 2 else -1.0) for x in xs]
    fit = ols_fit(xs, ys)
    near = ols_predict(9.5, xs, fit, 1.645)
    far = ols_predict(200.0, xs, fit, 1.645)
    assert (far['upper'] - far['lower']) > (near['upper'] - near['lower'])
    assert near['lower'] < near['yhat'] < near['upper']


def test_ewma_is_deterministic_and_bounded():
    series = [10.0, 12.0, 14.0, 9.0, 11.0]
    smoothed = ewma(series, 0.5)
    assert smoothed[0] == 10.0
    assert smoothed[1] == pytest.approx(11.0)
    assert all(min(series) <= value <= max(series) for value in smoothed)
    assert ewma(series, 1.0) == series


def test_linear_forecast_reports_horizon_baseline_and_interval():
    series = [float(3 * i + 1) for i in range(30)]
    result = linear_forecast(series, horizon=4, interval=0.9)
    assert result['backend'] == BACKEND
    assert result['points_used'] == 30
    assert len(result['predictions']) == 4
    assert result['predictions'][0]['x'] == 30.0
    assert result['predictions'][0]['yhat'] == pytest.approx(3 * 30 + 1, abs=1e-6)
    assert result['predictions'][0]['lower'] <= result['predictions'][0]['yhat'] <= result['predictions'][0]['upper']
    assert result['fit_window']['holdout'] >= 2
    assert 0 <= result['holdout']['mae'] < 1e-6
    assert result['holdout']['beats_baseline'] is True
    assert any('seasonality' in line for line in result['assumptions'])
    assert 'not evidence' in result['limitations']


def test_linear_forecast_flags_a_baseline_win_correctly():
    # A flat series: the last-value baseline is perfect, the trend cannot beat it.
    result = linear_forecast([5.0] * 20, horizon=2)
    assert result['holdout']['baseline_mae_last_value'] == 0.0
    assert result['holdout']['beats_baseline'] is False


def test_invalid_numeric_inputs_are_rejected():
    with pytest.raises(NumericInputError):
        linear_forecast([1.0, 2.0])
    with pytest.raises(NumericInputError):
        linear_forecast([1.0, float('nan'), 3.0, 4.0])
    with pytest.raises(NumericInputError):
        linear_forecast([1.0, 2.0, 3.0], horizon=0)
    with pytest.raises(NumericInputError):
        linear_forecast([1.0, 2.0, 3.0], interval=1.5)
    with pytest.raises(NumericInputError):
        linear_forecast([1.0, 2.0, 3.0], holdout_fraction=0.9)
    with pytest.raises(NumericInputError):
        ewma([1.0], 0)
    with pytest.raises(NumericInputError):
        mae([1.0], [1.0, 2.0])
    with pytest.raises(NumericInputError):
        rmse([], [])
    with pytest.raises(NumericInputError):
        ols_fit([1.0], [1.0])
