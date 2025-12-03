"""
Local regression dispersion fitting for DESeq2-like analysis.

This module provides alternative dispersion trend fitting methods
including local regression (LOWESS) and mean dispersion.

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
    - Cleveland WS (1979). Robust Locally Weighted Regression and Smoothing 
      Scatterplots. JASA 74:829-836
"""

import numpy as np
from scipy.special import polygamma
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


def fit_local_dispersion_trend(base_means, disp_gw, frac=0.2, it=3):
    """
    Fit dispersion trend using local regression (LOWESS).
    
    LOWESS (Locally Weighted Scatterplot Smoothing) is a non-parametric
    method that can capture complex relationships between mean expression
    and dispersion without assuming a specific functional form.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    frac : float, default 0.2
        Fraction of data to use in local fitting.
    it : int, default 3
        Number of robustness iterations.
        
    Returns
    -------
    callable
        Function that returns fitted dispersion for given mean values.
    np.ndarray
        Fitted dispersion values for input genes.
        
    Examples
    --------
    >>> trend_fn, fitted = fit_local_dispersion_trend(base_means, disp_gw)
    >>> new_disp = trend_fn(new_means)
    
    Notes
    -----
    LOWESS fitting is performed on log-log scale for better behavior
    across the wide range of mean expression values.
    
    The local trend may be preferred when:
    - The parametric (a/mean + b) trend doesn't fit well
    - There's unusual structure in the dispersion-mean relationship
    - You want to make minimal assumptions about functional form
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)
    
    # Filter for valid values
    mask = (base_means > 0) & (disp_gw > 1e-8) & (disp_gw < 100)
    mask &= np.isfinite(base_means) & np.isfinite(disp_gw)
    
    if mask.sum() < 10:
        # Not enough data, return constant function
        median_disp = np.median(disp_gw[disp_gw > 0])
        return lambda x: np.full_like(x, median_disp), np.full_like(disp_gw, median_disp)
    
    x = np.log10(base_means[mask])
    y = np.log10(disp_gw[mask])
    
    # Sort for LOWESS
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # Fit LOWESS
    smoothed = lowess(y_sorted, x_sorted, frac=frac, it=it, return_sorted=True)
    x_smooth = smoothed[:, 0]
    y_smooth = smoothed[:, 1]
    
    # Create interpolation function
    def trend_fn(means):
        means = np.asarray(means, dtype=float)
        log_means = np.log10(np.maximum(means, 1e-8))
        
        # Interpolate on log scale
        log_disp = np.interp(log_means, x_smooth, y_smooth,
                            left=y_smooth[0], right=y_smooth[-1])
        
        return 10 ** log_disp
    
    # Calculate fitted values for all genes
    fitted = np.full_like(disp_gw, np.nan)
    fitted[mask] = trend_fn(base_means[mask])
    fitted[~mask] = np.median(disp_gw[mask]) if mask.any() else 0.1
    
    return trend_fn, fitted


def fit_mean_dispersion(disp_gw, base_means=None, min_mean=0.0):
    """
    Fit a constant (mean) dispersion across all genes.
    
    This is useful for small datasets where there isn't enough
    information to estimate a mean-dispersion relationship.
    
    Parameters
    ----------
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    base_means : np.ndarray, optional
        Mean normalized counts (for filtering).
    min_mean : float, default 0.0
        Minimum mean for including gene in estimation.
        
    Returns
    -------
    callable
        Function that returns constant dispersion.
    float
        Mean dispersion value.
        
    Examples
    --------
    >>> trend_fn, mean_disp = fit_mean_dispersion(disp_gw)
    >>> # trend_fn(any_means) returns mean_disp
    
    Notes
    -----
    Using a mean dispersion is equivalent to assuming all genes have
    the same underlying dispersion, modulated only by sampling noise.
    This is a strong assumption but may be appropriate for:
    - Very small sample sizes (n < 4)
    - Highly homogeneous datasets
    - Quick exploratory analysis
    """
    disp_gw = np.asarray(disp_gw, dtype=float)
    
    # Filter valid values
    mask = (disp_gw > 1e-8) & (disp_gw < 100) & np.isfinite(disp_gw)
    
    if base_means is not None:
        base_means = np.asarray(base_means, dtype=float)
        mask &= base_means >= min_mean
    
    if mask.sum() == 0:
        mean_disp = 0.1
    else:
        # Use geometric mean for robustness
        mean_disp = np.exp(np.mean(np.log(disp_gw[mask])))
    
    def trend_fn(means):
        return np.full_like(np.asarray(means), mean_disp)
    
    return trend_fn, mean_disp


def estimate_prior_variance(base_means, disp_gw, disp_trend_fn, 
                            design_matrix=None, n_samples=None):
    """
    Estimate prior variance for dispersion shrinkage.
    
    The prior variance determines how much gene-wise dispersion
    estimates are shrunk toward the trend.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    disp_trend_fn : callable
        Fitted dispersion trend function.
    design_matrix : np.ndarray, optional
        Design matrix (for degrees of freedom calculation).
    n_samples : int, optional
        Number of samples (if design_matrix not provided).
        
    Returns
    -------
    float
        Estimated prior variance on log scale.
        
    Notes
    -----
    The prior variance is estimated from the residuals between
    gene-wise estimates and the trend, using a robust estimator
    (MAD scaled to match normal SD).
    
    This follows the approach in DESeq2 where the prior variance
    is estimated empirically from the data.
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)
    
    # Get trend values
    disp_trend = disp_trend_fn(base_means)
    
    # Calculate log residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        log_residuals = np.log(disp_gw) - np.log(disp_trend)
    
    # Filter valid values
    valid = np.isfinite(log_residuals) & (base_means > 1)
    
    if valid.sum() < 10:
        return 1.0
    
    # Robust estimate using MAD
    residuals = log_residuals[valid]
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma_prior = mad * 1.4826  # Scale MAD to match normal SD
    
    # Enforce minimum (following DESeq2)
    sigma_prior = max(sigma_prior, 0.25)
    
    return sigma_prior ** 2


def estimate_map_dispersions_local(base_means, disp_gw, disp_trend_fn,
                                   design_matrix, counts, size_factors,
                                   prior_var=None):
    """
    Estimate MAP dispersions using local trend.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    disp_trend_fn : callable
        Fitted dispersion trend function.
    design_matrix : np.ndarray
        Design matrix (samples x parameters).
    counts : np.ndarray
        Raw count matrix (genes x samples).
    size_factors : np.ndarray
        Size factors for each sample.
    prior_var : float, optional
        Prior variance. If None, estimated from data.
        
    Returns
    -------
    np.ndarray
        MAP dispersion estimates.
    np.ndarray
        Boolean mask of outlier genes.
        
    Notes
    -----
    MAP estimation shrinks gene-wise dispersion estimates toward
    the trend using Bayesian shrinkage with an empirically estimated
    prior.
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)
    
    # Get trend values
    disp_trend = disp_trend_fn(base_means)
    
    # Estimate prior variance if not provided
    if prior_var is None:
        prior_var = estimate_prior_variance(base_means, disp_gw, disp_trend_fn)
    
    sigma_prior = np.sqrt(prior_var)
    
    # Calculate degrees of freedom
    n_samples = counts.shape[1]
    n_params = design_matrix.shape[1]
    df = max(n_samples - n_params, 1)
    
    # Observed variance (from trigamma)
    var_obs = polygamma(1, df / 2.0)
    
    # Shrinkage weight
    weight = prior_var / (prior_var + var_obs)
    
    # MAP estimate on log scale
    log_map = weight * np.log(disp_gw) + (1.0 - weight) * np.log(disp_trend)
    disp_map = np.exp(log_map)
    
    # Identify outliers
    log_residuals = np.log(disp_gw) - np.log(disp_trend)
    is_outlier = np.abs(log_residuals / sigma_prior) > 2.0
    
    # For outliers, keep gene-wise estimate
    disp_final = disp_map.copy()
    disp_final[is_outlier] = disp_gw[is_outlier]
    
    return disp_final, is_outlier


def fit_dispersion_trend(base_means, disp_gw, fit_type='parametric', **kwargs):
    """
    Fit dispersion trend using specified method.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    fit_type : str, default 'parametric'
        Type of trend fitting:
        - 'parametric': a/mean + b (default DESeq2)
        - 'local': LOWESS smoothing
        - 'mean': Constant dispersion
    **kwargs
        Additional arguments passed to fitting function.
        
    Returns
    -------
    callable
        Trend function.
    np.ndarray or tuple
        Fitted values or coefficients.
    """
    if fit_type == 'parametric':
        from .dispersion_optimized import fit_parametric_dispersion_trend
        return fit_parametric_dispersion_trend(base_means, disp_gw)
    elif fit_type == 'local':
        return fit_local_dispersion_trend(base_means, disp_gw, **kwargs)
    elif fit_type == 'mean':
        trend_fn, mean_disp = fit_mean_dispersion(disp_gw, base_means)
        return trend_fn, (0.0, mean_disp)
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")
