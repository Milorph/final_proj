import numpy as np


def _safe_mean_var(x, axis=1):
    """
    Compute mean and unbiased variance along given axis.
    x: 2D array (genes x samples)
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis)
    # ddof=1 → unbiased variance; protect for n=1
    if x.shape[axis] > 1:
        var = x.var(axis=axis, ddof=1)
    else:
        var = np.zeros_like(mean)
    return mean, var


def estimate_gene_wise_dispersion(counts, size_factors, min_disp=1e-8):
    """
    Simplified gene-wise dispersion estimate using method of moments.

    Parameters
    ----------
    counts : array-like, shape (G, S)
        Raw count matrix (genes x samples).
    size_factors : array-like, shape (S,)
        Size factors per sample.
    min_disp : float
        Minimum dispersion for numerical stability.

    Returns
    -------
    base_means : (G,) array
        Mean normalized counts per gene.
    disp_gw : (G,) array
        Gene-wise dispersion estimates.
    """
    counts = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)

    if counts.ndim != 2:
        raise ValueError("counts must be 2D (genes x samples)")
    if sf.ndim != 1 or sf.shape[0] != counts.shape[1]:
        raise ValueError("size_factors must have length equal to number of samples")

    # normalize counts by size factors
    norm_counts = counts / sf

    # base mean and variance per gene
    base_means, var = _safe_mean_var(norm_counts, axis=1)

    # method-of-moments NB dispersion: Var(Y) = mu + alpha * mu^2
    # => alpha = max( (var - mean) / mean^2, min_disp )
    disp_gw = np.full_like(base_means, fill_value=min_disp)

    # only compute where mean > 0
    mask = base_means > 0
    numer = var[mask] - base_means[mask]
    denom = base_means[mask] ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = numer / denom

    # clip: avoid negative / tiny dispersions
    alpha = np.where(alpha > min_disp, alpha, min_disp)
    disp_gw[mask] = alpha

    return base_means, disp_gw


def fit_dispersion_trend(base_means, disp_gw, min_disp=1e-8):
    """
    Fit a simple parametric mean–dispersion relationship:
        alpha(mu) ≈ a / mu + b

    This is an approximation of DESeq2's parametric trend:
        disp ~ a1 / mu + a0

    Parameters
    ----------
    base_means : (G,) array
        Mean normalized counts per gene.
    disp_gw : (G,) array
        Gene-wise dispersion estimates.
    min_disp : float
        Minimum dispersion.

    Returns
    -------
    disp_trend_fn : callable
        Function mapping base_mean -> disp_trend.
    params : (a, b)
        Fitted parametric coefficients.
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)

    # use genes with reasonable dispersion above minimum
    use = (disp_gw > 10 * min_disp) & (base_means > 0)
    if use.sum() < 10:
        # fallback: constant dispersion
        mean_disp = np.mean(disp_gw[disp_gw > min_disp])
        def disp_trend_fn(mu):
            return np.full_like(np.asarray(mu, dtype=float), mean_disp)
        return disp_trend_fn, (0.0, mean_disp)

    mu = base_means[use]
    y = disp_gw[use]

    # linear model: y ≈ a / mu + b
    X = np.vstack([1.0 / mu, np.ones_like(mu)]).T
    # least squares fit
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta

    def disp_trend_fn(mu_query):
        mu_query = np.asarray(mu_query, dtype=float)
        mu_query = np.maximum(mu_query, 1e-8)
        val = a / mu_query + b
        # ensure non-negative and above min_disp
        return np.maximum(val, min_disp)

    return disp_trend_fn, (a, b)


def shrink_dispersion_map(base_means, disp_gw, disp_trend_fn,
                          outlier_sd=2.0, min_disp=1e-8):
    """
    Shrink gene-wise dispersions toward the trend in log-space.

    Simplified empirical Bayes:
        log alpha_MAP = 0.5 * (log alpha_gw + log alpha_trend)

    Outliers: genes whose log alpha_gw is far above trend are kept unshrunken.

    Parameters
    ----------
    base_means : (G,) array
    disp_gw : (G,) array
    disp_trend_fn : callable
        Output of fit_dispersion_trend.
    outlier_sd : float
        Threshold in SD units for putting genes into 'outlier' set.
    min_disp : float

    Returns
    -------
    disp_final : (G,) array
        Final dispersion per gene.
    disp_trend : (G,) array
        Trend dispersion per gene.
    disp_map : (G,) array
        Shrunken MAP dispersion per gene.
    is_outlier : (G,) bool array
        True for genes where we keep gene-wise dispersion.
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)

    disp_trend = disp_trend_fn(base_means)

    # log-space
    log_gw = np.log(np.maximum(disp_gw, min_disp))
    log_trend = np.log(np.maximum(disp_trend, min_disp))

    # estimate variability of log dispersions around trend
    resid = log_gw - log_trend
    resid_mean = np.mean(resid[np.isfinite(resid)])
    resid_sd = np.std(resid[np.isfinite(resid)], ddof=1) + 1e-8

    # simple shrinkage: average in log-space (equal weights)
    log_map = 0.5 * (log_gw + log_trend)
    disp_map = np.exp(log_map)

    # outliers: genes far above the trend
    is_outlier = (log_gw - log_trend) > (resid_mean + outlier_sd * resid_sd)
    is_outlier = np.where(np.isfinite(is_outlier), is_outlier, False)

    disp_final = disp_map.copy()
    disp_final[is_outlier] = disp_gw[is_outlier]

    # numeric safety
    disp_final = np.maximum(disp_final, min_disp)
    disp_map = np.maximum(disp_map, min_disp)

    return disp_final, disp_trend, disp_map, is_outlier


def estimate_dispersions(counts, size_factors,
                         min_disp=1e-8,
                         outlier_sd=2.0):
    """
    High-level convenience function:
    counts + size_factors -> final per-gene dispersions.

    This is the simplified analogue of:
        estimateDispersionsGeneEst + Fit + MAP

    Parameters
    ----------
    counts : (G, S) array
    size_factors : (S,) array
    min_disp : float
    outlier_sd : float

    Returns
    -------
    base_means : (G,) array
    disp_final : (G,) array
        Final dispersion used for testing.
    disp_gw : (G,) array
        Raw gene-wise dispersion estimates.
    disp_trend : (G,) array
        Trend dispersion as function of mean.
    disp_map : (G,) array
        MAP-shrunken dispersion.
    is_outlier : (G,) bool array
    """
    base_means, disp_gw = estimate_gene_wise_dispersion(
        counts, size_factors, min_disp=min_disp
    )
    disp_trend_fn, _ = fit_dispersion_trend(
        base_means, disp_gw, min_disp=min_disp
    )
    disp_final, disp_trend, disp_map, is_outlier = shrink_dispersion_map(
        base_means, disp_gw, disp_trend_fn,
        outlier_sd=outlier_sd, min_disp=min_disp
    )
    return base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier

