"""
Likelihood Ratio Test for differential expression analysis.

This module implements the likelihood ratio test (LRT) for comparing
nested models, which is useful for multi-factor designs and ANOVA-like
tests in RNA-seq differential expression analysis.

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
from scipy.stats import chi2
from scipy.special import gammaln
import statsmodels.api as sm


def nbinom_loglikelihood(y, mu, alpha):
    """
    Calculate negative binomial log-likelihood.
    
    Parameters
    ----------
    y : np.ndarray
        Observed counts.
    mu : np.ndarray
        Fitted mean values.
    alpha : float
        Dispersion parameter (variance = mu + alpha * mu^2).
        
    Returns
    -------
    float
        Log-likelihood value.
        
    Notes
    -----
    The negative binomial distribution parameterized by mean mu and
    dispersion alpha (variance = mu + alpha * mu^2):
    
    P(Y = y | mu, alpha) = Gamma(y + 1/alpha) / (Gamma(1/alpha) * y!) *
                           (1/(1 + alpha*mu))^(1/alpha) * 
                           (alpha*mu / (1 + alpha*mu))^y
    """
    alpha = max(alpha, 1e-10)
    r = 1.0 / alpha
    mu = np.maximum(mu, 1e-8)
    
    # Log-likelihood components
    prob = r / (r + mu)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) +
          r * np.log(prob) + y * np.log(1.0 - prob))
    
    return np.sum(ll)


def fit_nbinom_glm(y, X, alpha, offset=None):
    """
    Fit a negative binomial GLM and return log-likelihood.
    
    Parameters
    ----------
    y : np.ndarray
        Observed counts for one gene.
    X : np.ndarray
        Design matrix (samples x parameters).
    alpha : float
        Dispersion parameter.
    offset : np.ndarray, optional
        Offset term (log of size factors).
        
    Returns
    -------
    float
        Log-likelihood of the fitted model.
    np.ndarray
        Fitted mean values.
    np.ndarray
        Fitted coefficients.
    """
    if y.sum() == 0:
        return -np.inf, np.zeros_like(y), np.zeros(X.shape[1])
    
    alpha = max(alpha, 1e-10)
    
    try:
        fam = sm.families.NegativeBinomial(alpha=alpha)
        model = sm.GLM(y, X, family=fam, offset=offset)
        res = model.fit()
        
        mu = res.mu
        ll = nbinom_loglikelihood(y, mu, alpha)
        
        return ll, mu, res.params
    except Exception:
        # Fallback for convergence failures
        return -np.inf, np.zeros_like(y), np.zeros(X.shape[1])


def likelihood_ratio_test(counts, size_factors, dispersions, 
                          design_full, design_reduced):
    """
    Perform likelihood ratio test comparing full and reduced models.
    
    The LRT compares two nested models by computing:
    LRT = 2 * (logLik_full - logLik_reduced)
    
    Under the null hypothesis that the reduced model is adequate,
    LRT follows a chi-squared distribution with df equal to the
    difference in the number of parameters.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    size_factors : np.ndarray
        Size factors for each sample.
    dispersions : np.ndarray
        Gene-wise dispersion estimates.
    design_full : np.ndarray
        Full model design matrix (samples x p_full).
    design_reduced : np.ndarray
        Reduced model design matrix (samples x p_reduced).
        Must be nested within the full model (p_reduced < p_full).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'stat': LRT chi-squared statistics (genes,)
        - 'pvalue': P-values from chi-squared distribution (genes,)
        - 'df': Degrees of freedom for the test
        - 'logLik_full': Log-likelihoods for full model (genes,)
        - 'logLik_reduced': Log-likelihoods for reduced model (genes,)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Full model: ~ condition + batch
    >>> # Reduced model: ~ batch (tests effect of condition)
    >>> X_full = np.column_stack([np.ones(8), [0,0,1,1,0,0,1,1], [0,1,0,1,0,1,0,1]])
    >>> X_reduced = np.column_stack([np.ones(8), [0,1,0,1,0,1,0,1]])
    >>> result = likelihood_ratio_test(counts, sf, disp, X_full, X_reduced)
    
    Notes
    -----
    The LRT is particularly useful for:
    - Testing the significance of a factor with multiple levels
    - Multi-factor designs where you want to test one factor
      while controlling for others
    - Any situation where you want to compare nested models
    
    For simple two-group comparisons, the Wald test is usually sufficient
    and faster.
    
    References:
        Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
        and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
    """
    Y = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    disp = np.asarray(dispersions, dtype=float)
    X_full = np.asarray(design_full, dtype=float)
    X_reduced = np.asarray(design_reduced, dtype=float)
    
    G, S = Y.shape
    
    # Validate inputs
    if X_full.shape[0] != S or X_reduced.shape[0] != S:
        raise ValueError("Design matrices must have same number of rows as samples")
    
    p_full = X_full.shape[1]
    p_reduced = X_reduced.shape[1]
    
    if p_reduced >= p_full:
        raise ValueError("Reduced model must have fewer parameters than full model")
    
    df = p_full - p_reduced
    
    # Log of size factors for offset
    offset = np.log(sf + 1e-12)
    
    # Initialize output arrays
    ll_full = np.full(G, np.nan)
    ll_reduced = np.full(G, np.nan)
    lrt_stat = np.full(G, np.nan)
    pvals = np.ones(G)
    
    for g in range(G):
        y = Y[g, :]
        alpha = disp[g]
        
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.1
        
        # Fit full model
        ll_f, mu_f, _ = fit_nbinom_glm(y, X_full, alpha, offset)
        
        # Fit reduced model
        ll_r, mu_r, _ = fit_nbinom_glm(y, X_reduced, alpha, offset)
        
        ll_full[g] = ll_f
        ll_reduced[g] = ll_r
        
        # LRT statistic
        if np.isfinite(ll_f) and np.isfinite(ll_r):
            stat = 2 * (ll_f - ll_r)
            stat = max(0, stat)  # Should be non-negative
            lrt_stat[g] = stat
            pvals[g] = chi2.sf(stat, df)
    
    return {
        'stat': lrt_stat,
        'pvalue': pvals,
        'df': df,
        'logLik_full': ll_full,
        'logLik_reduced': ll_reduced
    }


def lrt_test(counts, size_factors, dispersions, design_full, design_reduced,
             alpha=0.05):
    """
    Simplified interface for likelihood ratio test with FDR correction.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    size_factors : np.ndarray
        Size factors for each sample.
    dispersions : np.ndarray
        Gene-wise dispersion estimates.
    design_full : np.ndarray
        Full model design matrix.
    design_reduced : np.ndarray
        Reduced model design matrix.
    alpha : float, default 0.05
        FDR threshold for significance.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'stat': LRT statistics
        - 'pvalue': Raw p-values
        - 'padj': BH-adjusted p-values
        - 'significant': Boolean array of significant genes
        
    Examples
    --------
    >>> result = lrt_test(counts, sf, disp, X_full, X_reduced, alpha=0.05)
    >>> sig_genes = np.where(result['significant'])[0]
    """
    from .nbinom_wald import benjamini_hochberg
    
    result = likelihood_ratio_test(counts, size_factors, dispersions,
                                   design_full, design_reduced)
    
    # BH correction
    pvals = result['pvalue']
    pvals_clean = np.where(np.isfinite(pvals), pvals, 1.0)
    padj = benjamini_hochberg(pvals_clean)
    
    return {
        'stat': result['stat'],
        'pvalue': pvals_clean,
        'padj': padj,
        'significant': padj < alpha,
        'df': result['df']
    }
