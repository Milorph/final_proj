import numpy as np
from scipy.special import polygamma


def _safe_mean_var(x, axis=1):
    """
    Compute mean and unbiased variance along given axis.
    x: 2D array (genes x samples)
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis)
    if x.shape[axis] > 1:
        var = x.var(axis=axis, ddof=1)
    else:
        var = np.zeros_like(mean)
    return mean, var


def estimate_gene_wise_dispersion(counts, size_factors, group_labels=None, min_disp=1e-8):
    """
    Gene-wise dispersion estimate with Pooled Method of Moments.
    """
    counts = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    norm_counts = counts / sf
    base_means = norm_counts.mean(axis=1)

    if group_labels is not None:
        groups = np.unique(group_labels)
        alpha_sum = np.zeros(counts.shape[0])
        weight_sum = np.zeros(counts.shape[0])
        
        for g in groups:
            mask_g = (np.asarray(group_labels) == g)
            n_g = np.sum(mask_g)
            if n_g <= 1: continue
            
            sub_counts = norm_counts[:, mask_g]
            mean_g, var_g = _safe_mean_var(sub_counts, axis=1)
            valid_g = mean_g > 1e-8
            
            alpha_g = np.full(counts.shape[0], min_disp)
            numer = var_g[valid_g] - mean_g[valid_g]
            denom = mean_g[valid_g] ** 2
            
            with np.errstate(divide="ignore", invalid="ignore"):
                a = numer / denom
            
            a = np.maximum(a, min_disp)
            alpha_g[valid_g] = a
            
            w = n_g - 1
            alpha_sum += alpha_g * w
            weight_sum += w
            
        disp_gw = np.full(counts.shape[0], min_disp)
        valid_w = weight_sum > 0
        disp_gw[valid_w] = alpha_sum[valid_w] / weight_sum[valid_w]
        return base_means, disp_gw
    else:
        # Fallback for no groups
        base_means, var = _safe_mean_var(norm_counts, axis=1)
        disp_gw = np.full_like(base_means, fill_value=min_disp)
        mask = base_means > 0
        numer = var[mask] - base_means[mask]
        denom = base_means[mask] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = numer / denom
        alpha = np.where(alpha > min_disp, alpha, min_disp)
        disp_gw[mask] = alpha
        return base_means, disp_gw


def fit_dispersion_trend(base_means, disp_gw, min_disp=1e-8):
    """Fit parametric trend: alpha = a/mu + b"""
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)

    use = (disp_gw > 10 * min_disp) & (base_means > 0)
    if use.sum() < 10:
        mean_disp = np.mean(disp_gw[disp_gw > min_disp])
        return lambda mu: np.full_like(mu, mean_disp), (0.0, mean_disp)

    mu = base_means[use]
    y = disp_gw[use]
    X = np.vstack([1.0 / mu, np.ones_like(mu)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta

    def disp_trend_fn(mu_query):
        mu_query = np.asarray(mu_query, dtype=float)
        mu_query = np.maximum(mu_query, 1e-8)
        val = a / mu_query + b
        return np.maximum(val, min_disp)

    return disp_trend_fn, (a, b)


def shrink_dispersion_map(base_means, disp_gw, disp_trend_fn, degrees_of_freedom,
                          outlier_sd=2.0, min_disp=1e-8):
    """
    Shrink gene-wise dispersions toward the trend using Empirical Bayes.
    
    New Logic:
    Calculates weights based on the theoretical variance of the Log-Dispersion estimator.
    Var(log_alpha_obs) approx Trigamma(df/2).
    """
    base_means = np.asarray(base_means, dtype=float)
    disp_gw = np.asarray(disp_gw, dtype=float)

    disp_trend = disp_trend_fn(base_means)

    # Log-space values
    log_gw = np.log(np.maximum(disp_gw, min_disp))
    log_trend = np.log(np.maximum(disp_trend, min_disp))

    # --- EMPIRICAL BAYES WEIGHTING ---
    
    # 1. Theoretical variance of the observation (likelihood width)
    # Based on the degrees of freedom available for estimating dispersion
    # Var(log_est) ~ polygamma(1, df/2)
    var_log_obs = polygamma(1, degrees_of_freedom / 2.0)
    
    # 2. Variance of the prior (how far true dispersions scatter around the trend)
    # We estimate this robustly from the residuals
    resid = log_gw - log_trend
    # Use interquartile range or robust std to ignore outliers
    q25, q75 = np.percentile(resid[np.isfinite(resid)], [25, 75])
    robust_std = (q75 - q25) / 1.349
    var_prior = robust_std ** 2
    
    # Ensure prior variance isn't zero
    var_prior = max(var_prior, 0.01)

    # 3. Calculate MAP Weight
    # Weight for Observation = Prior_Var / (Prior_Var + Obs_Var)
    # If Obs_Var is high (small samples), w_obs is small -> Strong Shrinkage
    w_obs = var_prior / (var_prior + var_log_obs)
    
    # Apply weighted average
    log_map = w_obs * log_gw + (1.0 - w_obs) * log_trend
    
    disp_map = np.exp(log_map)

    # --- Outlier Detection ---
    resid_mean = np.mean(resid[np.isfinite(resid)])
    is_outlier = (log_gw - log_trend) > (resid_mean + outlier_sd * robust_std)
    is_outlier = np.where(np.isfinite(is_outlier), is_outlier, False)

    disp_final = disp_map.copy()
    disp_final[is_outlier] = disp_gw[is_outlier]
    disp_final = np.maximum(disp_final, min_disp)

    return disp_final, disp_trend, disp_map, is_outlier


def estimate_dispersions(counts, size_factors,
                         group_labels=None,
                         degrees_of_freedom=None,
                         min_disp=1e-8,
                         outlier_sd=2.0):
    """
    High-level convenience function.
    """
    base_means, disp_gw = estimate_gene_wise_dispersion(
        counts, size_factors, group_labels=group_labels, min_disp=min_disp
    )
    disp_trend_fn, _ = fit_dispersion_trend(
        base_means, disp_gw, min_disp=min_disp
    )
    
    # If df not provided, assume N - 1 (conservative)
    if degrees_of_freedom is None:
        degrees_of_freedom = counts.shape[1] - 1

    disp_final, disp_trend, disp_map, is_outlier = shrink_dispersion_map(
        base_means, disp_gw, disp_trend_fn, 
        degrees_of_freedom=degrees_of_freedom,
        outlier_sd=outlier_sd, min_disp=min_disp
    )
    return base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier