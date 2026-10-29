! HELIX native numerical kernels (Fortran 90, built with f2py).
!
! These routines power HELIX's data-driven prediction tools: ordinary least
! squares, autoregressive (AR) forecasting, multiple linear regression,
! polynomial trend fitting, EWMA smoothing and error metrics. They are
! numerically identical to the pure-Python fallback in helix/numerics.py and
! are only used when a compiled module is available. The language model itself
! is not written in Fortran and gains nothing from being written in Fortran.
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


! ------------------------------------------------------------------
! Autoregressive (AR(p)) forecasting: predict future values from the
! series' own past. This is the "self-generating replies from data"
! kernel -- the model extrapolates a series using only its own history.
! ------------------------------------------------------------------

subroutine ar_fit(x, n, p, phi, intercept, residual_sd)
    ! Fit x_t = intercept + phi_1 x_{t-1} + ... + phi_p x_{t-p} by OLS.
    implicit none
    integer, intent(in) :: n, p
    double precision, intent(in) :: x(n)
    double precision, intent(out) :: phi(p), intercept, residual_sd
    double precision :: design(100, 12), target(100)
    double precision :: xtx(12, 12), xty(12), work(12)
    double precision :: mean_x, sse, resid, pred
    integer :: i, j, k, m, nrows, info
    integer :: ipiv(12)
    double precision, parameter :: tiny = 1.0d-12

    phi = 0.0d0
    intercept = 0.0d0
    residual_sd = 0.0d0
    if (n <= p + 1 .or. p < 1 .or. p > 12) return

    nrows = n - p
    ! Build design matrix: [1, x_{t-1}, ..., x_{t-p}] for t = p+1 .. n
    do i = 1, nrows
        design(i, 1) = 1.0d0
        do j = 1, p
            design(i, j + 1) = x(p + i - j)  ! x_{t-j}
        end do
        target(i) = x(p + i)
    end do

    ! Normal equations: X^T X beta = X^T y
    xtx = 0.0d0
    xty = 0.0d0
    do i = 1, nrows
        do j = 1, p + 1
            xty(j) = xty(j) + design(i, j) * target(i)
            do k = 1, p + 1
                xtx(j, k) = xtx(j, k) + design(i, j) * design(i, k)
            end do
        end do
    end do

    ! Solve via Gaussian elimination with partial pivoting (small fixed system).
    call _solve_linear(xtx, p + 1, xty, work, ipiv, info)
    if (info /= 0) then
        intercept = sum(x) / dble(n)
        return
    end if

    intercept = work(1)
    do j = 1, p
        phi(j) = work(j + 1)
    end do

    sse = 0.0d0
    do i = 1, nrows
        pred = intercept
        do j = 1, p
            pred = pred + phi(j) * x(p + i - j)
        end do
        resid = target(i) - pred
        sse = sse + resid * resid
    end do
    m = nrows - (p + 1)
    if (m > 0) then
        residual_sd = sqrt(sse / dble(m))
    else
        residual_sd = 0.0d0
    end if
end subroutine ar_fit


subroutine ar_predict(x, n, p, phi, intercept, steps, out)
    ! Iteratively forecast `steps` future values using the fitted AR(p).
    implicit none
    integer, intent(in) :: n, p, steps
    double precision, intent(in) :: x(n), phi(p), intercept
    double precision, intent(out) :: out(steps)
    double precision :: buf(200)
    integer :: i, j, t, len

    len = n
    do i = 1, n
        buf(i) = x(i)
    end do
    do t = 1, steps
        out(t) = intercept
        do j = 1, p
            out(t) = out(t) + phi(j) * buf(len - j + 1)
        end do
        len = len + 1
        buf(len) = out(t)
    end do
end subroutine ar_predict


! ------------------------------------------------------------------
! Multiple linear regression: y = intercept + beta_1 x_1 + ... + beta_k x_k.
! ------------------------------------------------------------------

subroutine mlr_fit(X, y, n, k, beta, intercept, residual_sd, r2)
    implicit none
    integer, intent(in) :: n, k
    double precision, intent(in) :: X(n, k), y(n)
    double precision, intent(out) :: beta(k), intercept, residual_sd, r2
    double precision :: xtx(13, 13), xty(13), work(13)
    double precision :: mean_y, sse, resid, pred, sst
    integer :: i, j, m, info, ncol
    integer :: ipiv(13)

    beta = 0.0d0
    intercept = 0.0d0
    residual_sd = 0.0d0
    r2 = 0.0d0
    if (n < k + 2 .or. k < 1 .or. k > 12) return

    ncol = k + 1
    xtx = 0.0d0
    xty = 0.0d0
    do i = 1, n
        xty(1) = xty(1) + y(i)
        do j = 1, k
            xty(j + 1) = xty(j + 1) + X(i, j) * y(i)
        end do
    end do
    do i = 1, n
        xtx(1, 1) = xtx(1, 1) + 1.0d0
        do j = 1, k
            xtx(1, j + 1) = xtx(1, j + 1) + X(i, j)
            xtx(j + 1, 1) = xtx(j + 1, 1) + X(i, j)
            do m = 1, k
                xtx(j + 1, m + 1) = xtx(j + 1, m + 1) + X(i, j) * X(i, m)
            end do
        end do
    end do

    call _solve_linear(xtx, ncol, xty, work, ipiv, info)
    if (info /= 0) then
        intercept = sum(y) / dble(n)
        return
    end if

    intercept = work(1)
    do j = 1, k
        beta(j) = work(j + 1)
    end do

    sse = 0.0d0
    sst = 0.0d0
    mean_y = sum(y) / dble(n)
    do i = 1, n
        pred = intercept
        do j = 1, k
            pred = pred + beta(j) * X(i, j)
        end do
        resid = y(i) - pred
        sse = sse + resid * resid
        sst = sst + (y(i) - mean_y) ** 2
    end do
    m = n - ncol
    if (m > 0) then
        residual_sd = sqrt(sse / dble(m))
    else
        residual_sd = 0.0d0
    end if
    if (sst > 0.0d0) then
        r2 = 1.0d0 - sse / sst
    else
        r2 = 0.0d0
    end if
end subroutine mlr_fit


subroutine mlr_predict(x_row, k, beta, intercept, yhat)
    implicit none
    integer, intent(in) :: k
    double precision, intent(in) :: x_row(k), beta(k), intercept
    double precision, intent(out) :: yhat
    integer :: j
    yhat = intercept
    do j = 1, k
        yhat = yhat + beta(j) * x_row(j)
    end do
end subroutine mlr_predict


! ------------------------------------------------------------------
! Polynomial trend fit: y = c_0 + c_1 t + ... + c_d t^d (least squares).
! ------------------------------------------------------------------

subroutine poly_fit(y, n, d, coef, residual_sd)
    implicit none
    integer, intent(in) :: n, d
    double precision, intent(in) :: y(n)
    double precision, intent(out) :: coef(d + 1), residual_sd
    double precision :: xtx(13, 13), xty(13), work(13)
    double precision :: sse, resid, pred, tpow
    integer :: i, j, m, info, ncol
    integer :: ipiv(13)

    coef = 0.0d0
    residual_sd = 0.0d0
    if (n < d + 2 .or. d < 1 .or. d > 12) return

    ncol = d + 1
    xtx = 0.0d0
    xty = 0.0d0
    do i = 1, n
        tpow = 1.0d0
        do j = 0, d
            xty(j + 1) = xty(j + 1) + tpow * y(i)
            do m = 0, d
                xtx(j + 1, m + 1) = xtx(j + 1, m + 1) + tpow * (i ** m)
            end do
            tpow = tpow * dble(i)
        end do
    end do

    call _solve_linear(xtx, ncol, xty, work, ipiv, info)
    if (info /= 0) then
        coef(1) = sum(y) / dble(n)
        return
    end if
    do j = 0, d
        coef(j + 1) = work(j + 1)
    end do

    sse = 0.0d0
    do i = 1, n
        pred = 0.0d0
        tpow = 1.0d0
        do j = 0, d
            pred = pred + coef(j + 1) * tpow
            tpow = tpow * dble(i)
        end do
        resid = y(i) - pred
        sse = sse + resid * resid
    end do
    m = n - ncol
    if (m > 0) then
        residual_sd = sqrt(sse / dble(m))
    else
        residual_sd = 0.0d0
    end if
end subroutine poly_fit


subroutine poly_predict(d, coef, t, yhat)
    implicit none
    integer, intent(in) :: d
    double precision, intent(in) :: coef(d + 1), t
    double precision, intent(out) :: yhat
    double precision :: tpow
    integer :: j
    yhat = 0.0d0
    tpow = 1.0d0
    do j = 0, d
        yhat = yhat + coef(j + 1) * tpow
        tpow = tpow * t
    end do
end subroutine poly_predict


! ------------------------------------------------------------------
! Pearson correlation matrix of k series, each of length n.
! ------------------------------------------------------------------

subroutine correlation(X, n, k, corr)
    implicit none
    integer, intent(in) :: n, k
    double precision, intent(in) :: X(n, k)
    double precision, intent(out) :: corr(k, k)
    double precision :: mean(13), sd(13)
    double precision :: cov
    integer :: i, j, m
    double precision, parameter :: eps = 1.0d-12

    do j = 1, k
        mean(j) = 0.0d0
        do i = 1, n
            mean(j) = mean(j) + X(i, j)
        end do
        mean(j) = mean(j) / dble(n)
    end do

    do j = 1, k
        sd(j) = 0.0d0
        do i = 1, n
            sd(j) = sd(j) + (X(i, j) - mean(j)) ** 2
        end do
        sd(j) = sqrt(sd(j) / dble(max(n - 1, 1)))
    end do

    do j = 1, k
        corr(j, j) = 1.0d0
        do m = j + 1, k
            cov = 0.0d0
            do i = 1, n
                cov = cov + (X(i, j) - mean(j)) * (X(i, m) - mean(m))
            end do
            cov = cov / dble(max(n - 1, 1))
            if (sd(j) > eps .and. sd(m) > eps) then
                corr(j, m) = cov / (sd(j) * sd(m))
            else
                corr(j, m) = 0.0d0
            end if
            corr(m, j) = corr(j, m)
        end do
    end do
end subroutine correlation


! ------------------------------------------------------------------
! Small dense linear solver: A (n x n) x = b, with partial pivoting.
! ------------------------------------------------------------------

subroutine _solve_linear(A, n, b, x, ipiv, info)
    implicit none
    integer, intent(in) :: n
    double precision, intent(inout) :: A(n, n), b(n)
    double precision, intent(out) :: x(n)
    integer, intent(out) :: ipiv(n), info
    double precision :: work(13), maxval, pivot, factor
    integer :: i, j, k, pivrow

    info = 0
    do i = 1, n
        ipiv(i) = i
    end do

    ! LU with partial pivoting (in place).
    do k = 1, n - 1
        pivrow = k
        maxval = abs(A(k, k))
        do i = k + 1, n
            if (abs(A(i, k)) > maxval) then
                maxval = abs(A(i, k))
                pivrow = i
            end do
        end do
        if (maxval < 1.0d-14) then
            info = k
            return
        end if
        if (pivrow /= k) then
            do j = 1, n
                work(j) = A(k, j)
                A(k, j) = A(pivrow, j)
                A(pivrow, j) = work(j)
            end do
            work(1) = b(k)
            b(k) = b(pivrow)
            b(pivrow) = work(1)
            i = ipiv(k)
            ipiv(k) = ipiv(pivrow)
            ipiv(pivrow) = i
        end if
        pivot = A(k, k)
        do i = k + 1, n
            factor = A(i, k) / pivot
            A(i, k) = factor
            do j = k + 1, n
                A(i, j) = A(i, j) - factor * A(k, j)
            end do
            b(i) = b(i) - factor * b(k)
        end do
    end do

    if (abs(A(n, n)) < 1.0d-14) then
        info = n
        return
    end if

    ! Forward substitution: L y = b.
    do i = 2, n
        do j = 1, i - 1
            b(i) = b(i) - A(i, j) * b(j)
        end do
    end do

    ! Back substitution: U x = y.
    do i = n, 1, -1
        x(i) = b(i)
        do j = i + 1, n
            x(i) = x(i) - A(i, j) * x(j)
        end do
        x(i) = x(i) / A(i, i)
    end do
end subroutine _solve_linear
