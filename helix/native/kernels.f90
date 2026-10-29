! HELIX native numerical kernels (Fortran 90, built with f2py).
!
! These routines cover one narrow, measurable job: ordinary least squares with a
! prediction interval, plus an exponentially weighted moving average. They are
! numerically identical to the pure-Python fallback in helix/numerics.py and are
! only used when a compiled module is available. The language model itself is
! not written in Fortran and gains nothing from being written in Fortran.
!
! Build:  python helix/native/build_fortran.py
! Test:   python -m pytest tests/test_numerics.py

subroutine ols_fit(x, y, n, slope, intercept, residual_sd, r2)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: x(n), y(n)
    double precision, intent(out) :: slope, intercept, residual_sd, r2
    double precision :: mean_x, mean_y, sxx, sxy, syy, sse, sst, resid
    integer :: i

    slope = 0.0d0
    intercept = 0.0d0
    residual_sd = 0.0d0
    r2 = 0.0d0
    if (n < 2) return

    mean_x = sum(x) / dble(n)
    mean_y = sum(y) / dble(n)
    sxx = 0.0d0
    sxy = 0.0d0
    syy = 0.0d0
    do i = 1, n
        sxx = sxx + (x(i) - mean_x) ** 2
        sxy = sxy + (x(i) - mean_x) * (y(i) - mean_y)
        syy = syy + (y(i) - mean_y) ** 2
    end do
    if (sxx <= 0.0d0) then
        intercept = mean_y
        return
    end if
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    sse = 0.0d0
    do i = 1, n
        resid = y(i) - (intercept + slope * x(i))
        sse = sse + resid * resid
    end do
    sst = syy
    if (n > 2) then
        residual_sd = sqrt(sse / dble(n - 2))
    else
        residual_sd = 0.0d0
    end if
    if (sst > 0.0d0) then
        r2 = 1.0d0 - sse / sst
    else
        r2 = 0.0d0
    end if
end subroutine ols_fit


subroutine ols_predict(x0, n, mean_x, sxx, slope, intercept, residual_sd, z, yhat, lower, upper)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: x0, mean_x, sxx, slope, intercept, residual_sd, z
    double precision, intent(out) :: yhat, lower, upper
    double precision :: margin

    yhat = intercept + slope * x0
    margin = 0.0d0
    if (sxx > 0.0d0 .and. n > 2) then
        margin = z * residual_sd * sqrt(1.0d0 + 1.0d0 / dble(n) + ((x0 - mean_x) ** 2) / sxx)
    end if
    lower = yhat - margin
    upper = yhat + margin
end subroutine ols_predict


subroutine ewma(x, n, alpha, out)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: x(n), alpha
    double precision, intent(out) :: out(n)
    integer :: i

    if (n < 1) return
    out(1) = x(1)
    do i = 2, n
        out(i) = alpha * x(i) + (1.0d0 - alpha) * out(i - 1)
    end do
end subroutine ewma


subroutine rmse(x, y, n, value)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: x(n), y(n)
    double precision, intent(out) :: value
    integer :: i
    value = 0.0d0
    if (n < 1) return
    do i = 1, n
        value = value + (x(i) - y(i)) ** 2
    end do
    value = sqrt(value / dble(n))
end subroutine rmse
