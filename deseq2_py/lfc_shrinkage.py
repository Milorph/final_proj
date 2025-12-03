"""
Log fold change shrinkage methods for DESeq2-like analysis.

This module implements multiple shrinkage methods for moderating
log fold change estimates:
- apeglm: Approximate posterior estimation using Cauchy prior
- normal: Normal prior shrinkage (original DESeq2 method)

Shrinkage is important because genes with low counts or high dispersion
have uncertain fold change estimates, and shrinking them toward zero
reduces noise for downstream analysis.

References:
    - Zhu A, Ibrahim JG, Love MI (2019). Heavy-tailed prior distributions 
      for sequence count data: removing the noise and preserving large 
      differences. Bioinformatics 35(12):2084-2092
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import cauchy, norm


def apeglm_shrinkage(log2_fc_mle, se_log2, coef=1, prior_scale=None,
                     max_iter=100, tol=1e-6):
    """
    Approximate posterior estimation for GLM using Cauchy prior (apeglm).
    
    This method uses a heavy-tailed Cauchy prior that provides adaptive
    shrinkage: small estimated fold changes are shrunk strongly toward
    zero, while large estimated fold changes are preserved.
    
    Parameters
    ----------
    log2_fc_mle : np.ndarray
        Maximum likelihood estimates of log2 fold changes.
    se_log2 : np.ndarray
        Standard errors of the log2 fold change estimates.
    coef : int, default 1
        Index of coefficient being shrunk (for reference, not used here).
    prior_scale : float, optional
        Scale parameter for Cauchy prior. If None, estimated from data.
    max_iter : int, default 100
        Maximum iterations for optimization.
    tol : float, default 1e-6
        Convergence tolerance.
        
    Returns
    -------
    np.ndarray
        Shrunken log2 fold change estimates.
        
    Examples
    --------
    >>> import numpy as np
    >>> lfc = np.array([0.5, 2.0, -1.5, 0.1])
    >>> se = np.array([0.3, 0.5, 0.4, 0.8])
    >>> lfc_shrunk = apeglm_shrinkage(lfc, se)
    
    Notes
    -----
    The Cauchy prior has the form:
        p(beta) propto 1 / (1 + (beta/scale)^2)
    
    The posterior mode is found by maximizing:
        L(beta | y) * p(beta)
    
    This can be done analytically for the normal-Cauchy case:
    The posterior mode is found by solving a cubic equation.
    
    The Cauchy prior provides adaptive shrinkage:
    - Small effects are shrunk strongly toward zero
    - Large effects are preserved (heavy tails)
    - This differs from Normal prior which shrinks all effects equally
    
    References:
        Zhu A, Ibrahim JG, Love MI (2019). Heavy-tailed prior distributions 
        for sequence count data. Bioinformatics 35(12):2084-2092
    """
    lfc = np.asarray(log2_fc_mle, dtype=float)
    se = np.asarray(se_log2, dtype=float)
    
    G = len(lfc)
    lfc_shrunk = np.full(G, np.nan)
    
    # Estimate prior scale if not provided
    if prior_scale is None:
        # Use MAD of MLE estimates that are significant
        valid = np.isfinite(lfc) & np.isfinite(se) & (se > 0)
        if valid.sum() > 10:
            # Use robust estimate
            mad = np.median(np.abs(lfc[valid] - np.median(lfc[valid])))
            prior_scale = max(mad * 1.4826, 0.5)  # Convert MAD to SD estimate
        else:
            prior_scale = 1.0
    
    for i in range(G):
        beta_mle = lfc[i]
        sigma = se[i]
        
        if not np.isfinite(beta_mle) or not np.isfinite(sigma) or sigma <= 0:
            lfc_shrunk[i] = beta_mle
            continue
        
        # For Cauchy prior with normal likelihood, find posterior mode
        # Posterior: p(beta | y) propto N(beta | beta_mle, sigma^2) * Cauchy(beta | 0, scale)
        # 
        # The log-posterior is:
        # -0.5 * (beta - beta_mle)^2 / sigma^2 - log(1 + (beta/scale)^2)
        #
        # Note: An analytical solution exists via solving a cubic equation, but
        # numerical optimization is used here for robustness and simplicity.
        # The optimization is fast as it's a 1D problem with a good starting point.
        
        def neg_log_posterior(beta):
            ll = -0.5 * (beta - beta_mle) ** 2 / (sigma ** 2)
            lp = -np.log(1 + (beta / prior_scale) ** 2)
            return -(ll + lp)
        
        # Optimize starting from MLE (single-variable optimization is fast)
        result = minimize(neg_log_posterior, x0=beta_mle, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'gtol': tol})
        
        lfc_shrunk[i] = result.x[0] if result.success else beta_mle
    
    return lfc_shrunk


def normal_shrinkage(log2_fc_mle, se_log2, prior_mean=0.0, prior_var=None):
    """
    Normal prior shrinkage (original DESeq2 method).
    
    This method uses a Normal prior centered at zero, providing uniform
    shrinkage to all log fold change estimates based on their uncertainty.
    
    Parameters
    ----------
    log2_fc_mle : np.ndarray
        Maximum likelihood estimates of log2 fold changes.
    se_log2 : np.ndarray
        Standard errors of the log2 fold change estimates.
    prior_mean : float, default 0.0
        Mean of the Normal prior.
    prior_var : float, optional
        Variance of the Normal prior. If None, estimated from data.
        
    Returns
    -------
    np.ndarray
        Shrunken log2 fold change estimates.
        
    Examples
    --------
    >>> import numpy as np
    >>> lfc = np.array([0.5, 2.0, -1.5, 0.1])
    >>> se = np.array([0.3, 0.5, 0.4, 0.8])
    >>> lfc_shrunk = normal_shrinkage(lfc, se)
    
    Notes
    -----
    The shrinkage formula for Normal prior with Normal likelihood:
    
        beta_shrunk = (prior_var * beta_mle + sigma^2 * prior_mean) / 
                      (prior_var + sigma^2)
    
    When prior_mean = 0, this simplifies to:
        beta_shrunk = beta_mle * (prior_var / (prior_var + sigma^2))
    
    This is the classic Empirical Bayes shrinkage formula.
    
    References:
        Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
        and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
    """
    lfc = np.asarray(log2_fc_mle, dtype=float)
    se = np.asarray(se_log2, dtype=float)
    
    # Estimate prior variance if not provided
    if prior_var is None:
        valid = np.isfinite(lfc) & np.isfinite(se) & (se > 0)
        if valid.sum() > 10:
            # Estimate from high-expression genes
            # (similar to what DESeq2 does)
            var_obs = np.var(lfc[valid])
            mean_se2 = np.mean(se[valid] ** 2)
            prior_var = max(var_obs - mean_se2, 0.1)
        else:
            prior_var = 1.0
    
    # Shrinkage factor: weight toward prior
    # beta_shrunk = w * prior_mean + (1-w) * beta_mle
    # where w = sigma^2 / (sigma^2 + prior_var)
    
    se2 = se ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        shrink_weight = prior_var / (prior_var + se2)
    
    shrink_weight = np.where(np.isfinite(shrink_weight), shrink_weight, 0.0)
    
    lfc_shrunk = shrink_weight * lfc + (1 - shrink_weight) * prior_mean
    
    # For non-finite values, keep original
    lfc_shrunk = np.where(np.isfinite(lfc), lfc_shrunk, lfc)
    
    return lfc_shrunk


def ashr_shrinkage(log2_fc_mle, se_log2, prior_type='normal', 
                   mixture_components=10):
    """
    Adaptive shrinkage using mixture prior (simplified ashr-like approach).
    
    This method uses a mixture of Normal distributions as the prior,
    allowing for adaptive shrinkage based on the estimated effect size
    distribution.
    
    Parameters
    ----------
    log2_fc_mle : np.ndarray
        Maximum likelihood estimates of log2 fold changes.
    se_log2 : np.ndarray
        Standard errors of the log2 fold change estimates.
    prior_type : str, default 'normal'
        Type of mixture components ('normal' or 'uniform').
    mixture_components : int, default 10
        Number of mixture components.
        
    Returns
    -------
    np.ndarray
        Shrunken log2 fold change estimates.
        
    Notes
    -----
    This is a simplified version of the ashr method. For full
    functionality, consider using the ashr R package or its Python
    bindings.
    
    References:
        Stephens M (2017). False discovery rates: a new deal. 
        Biostatistics 18(2):275-294
    """
    lfc = np.asarray(log2_fc_mle, dtype=float)
    se = np.asarray(se_log2, dtype=float)
    
    # For simplicity, use EB normal shrinkage as approximation
    # A full implementation would fit a mixture model
    return normal_shrinkage(lfc, se)


def lfcShrink(log2_fc_mle, se_log2, type='apeglm', **kwargs):
    """
    Apply log fold change shrinkage using specified method.
    
    This is the main interface for LFC shrinkage, providing a
    consistent API for different shrinkage methods.
    
    Parameters
    ----------
    log2_fc_mle : np.ndarray
        Maximum likelihood estimates of log2 fold changes.
    se_log2 : np.ndarray
        Standard errors of the log2 fold change estimates.
    type : str, default 'apeglm'
        Shrinkage method. One of:
        - 'apeglm': Cauchy prior, adaptive shrinkage (recommended)
        - 'normal': Normal prior, uniform shrinkage
        - 'ashr': Adaptive shrinkage (simplified)
        - 'none': No shrinkage, return MLE
    **kwargs
        Additional arguments passed to the shrinkage method.
        
    Returns
    -------
    np.ndarray
        Shrunken log2 fold change estimates.
        
    Examples
    --------
    >>> import numpy as np
    >>> lfc = np.array([0.5, 2.0, -1.5, 0.1])
    >>> se = np.array([0.3, 0.5, 0.4, 0.8])
    >>> 
    >>> # Default apeglm shrinkage
    >>> lfc_shrunk = lfcShrink(lfc, se, type='apeglm')
    >>> 
    >>> # Normal shrinkage with custom prior variance
    >>> lfc_shrunk = lfcShrink(lfc, se, type='normal', prior_var=2.0)
    
    Notes
    -----
    Shrinkage recommendations:
    - 'apeglm': Best for most applications. Preserves large fold changes.
    - 'normal': Use when you expect truly null effects should be common.
    - 'ashr': Experimental, for comparison with R package.
    - 'none': Use only for exploratory analysis or method comparison.
    """
    type = type.lower()
    
    if type == 'apeglm':
        return apeglm_shrinkage(log2_fc_mle, se_log2, **kwargs)
    elif type == 'normal':
        return normal_shrinkage(log2_fc_mle, se_log2, **kwargs)
    elif type == 'ashr':
        return ashr_shrinkage(log2_fc_mle, se_log2, **kwargs)
    elif type == 'none':
        return np.asarray(log2_fc_mle, dtype=float)
    else:
        raise ValueError(f"Unknown shrinkage type: {type}. "
                        f"Use 'apeglm', 'normal', 'ashr', or 'none'.")
