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


def estimate_gene_wise_dispersion(counts, size_factors, group_labels=None, min_disp=1e-8):
    """
    Gene-wise dispersion estimate.
    
    If group_labels is provided, it calculates "Pooled Method of Moments":
      - Calculate Variance within each group.
      - Pool the alpha estimates (weighted by degrees of freedom).
    
    This prevents DE genes (high variance across all samples) from having 
    artificially inflated dispersion.
    """
    counts = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)

    if counts.ndim != 2:
        raise ValueError("counts must be 2D (genes x samples)")
    if sf.ndim != 1 or sf.shape[0] != counts.shape[1]:
        raise ValueError("size_factors must have length equal to number of samples")

    # normalize counts by size factors
    norm_counts = counts / sf
    
    # We always need global base_means for the trend fit later
    base_means = norm_counts.mean(axis=1)

    if group_labels is not None:
        # --- POOLED ESTIMATION (Fix for "No Red Dots") ---
        groups = np.unique(group_labels)
        alpha_sum = np.zeros(counts.shape[0])
        weight_sum = np.zeros(counts.shape[0])
        
        # We will calculate alpha for each group and average them
        for g in groups:
            mask_g = (np.asarray(group_labels) == g)
            n_g = np.sum(mask_g)
            
            if n_g <= 1: 
                # Cannot estimate variance from 1 sample, skip contribution
                continue
            
            # Slice data for this group
            sub_counts = norm_counts[:, mask_g]
            mean_g, var_g = _safe_mean_var(sub_counts, axis=1)
            
            # Alpha = (Var - Mean) / Mean^2
            # Only valid where mean > 0
            valid_g = mean_g > 1e-8
            
            # Calculate alpha for this group
            alpha_g = np.full(counts.shape[0], min_disp)
            
            numer = var_g[valid_g] - mean_g[valid_g]
            denom = mean_g[valid_g] ** 2
            
            with np.errstate(divide="ignore", invalid="ignore"):
                a = numer / denom
            
            # Clip
            a = np.maximum(a, min_disp)
            alpha_g[valid_g] = a
            
            # Add to pool (weighted by N-1)
            w = n_g - 1
            alpha_sum += alpha_g * w
            weight_sum += w
            
        # Final weighted average
        disp_gw = np.full(counts.shape[0], min_disp)
        valid_w = weight_sum > 0
        disp_gw[valid_w] = alpha_sum[valid_w] / weight_sum[valid_w]
        
        return base_means, disp_gw

    else:
        # --- GLOBAL ESTIMATION (Original / Conservative) ---
        # Assumes Intercept-only model (Mean is same for all samples)
        
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
                         group_labels=None,
                         min_disp=1e-8,
                         outlier_sd=2.0):
    """
    High-level convenience function with Group Awareness.
    """
    base_means, disp_gw = estimate_gene_wise_dispersion(
        counts, size_factors, group_labels=group_labels, min_disp=min_disp
    )
    disp_trend_fn, _ = fit_dispersion_trend(
        base_means, disp_gw, min_disp=min_disp
    )
    disp_final, disp_trend, disp_map, is_outlier = shrink_dispersion_map(
        base_means, disp_gw, disp_trend_fn,
        outlier_sd=outlier_sd, min_disp=min_disp
    )
    return base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier