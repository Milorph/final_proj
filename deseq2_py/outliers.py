"""
Cook's distance and outlier detection for DESeq2-like analysis.

This module provides functions for detecting outlier samples in
RNA-seq differential expression analysis using Cook's distance.

References:
    - Cook RD (1977). Detection of Influential Observation in Linear 
      Regression. Technometrics 19:15-18
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
from scipy.stats import f as f_dist
import statsmodels.api as sm


def calculate_cooks_distance(y, X, mu, disp, offset=None):
    """
    Calculate Cook's distance for each observation.
    
    Cook's distance measures the influence of each observation on the
    fitted values. Large values indicate observations that have a
    disproportionate effect on the model fit.
    
    Parameters
    ----------
    y : np.ndarray
        Observed counts (samples,).
    X : np.ndarray
        Design matrix (samples x parameters).
    mu : np.ndarray
        Fitted mean values (samples,).
    disp : float
        Dispersion parameter.
    offset : np.ndarray, optional
        Offset term (log of size factors).
        
    Returns
    -------
    np.ndarray
        Cook's distance for each observation (samples,).
        
    Notes
    -----
    Cook's distance is calculated as:
    
        D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
    
    Where:
    - e_i is the Pearson residual for observation i
    - p is the number of parameters
    - MSE is the mean squared error
    - h_ii is the leverage (diagonal of hat matrix)
    
    For GLMs with non-constant variance, we use the formula:
        D_i = (e_i^2 * h_ii) / (p * (1 - h_ii)^2)
    
    Where e_i is the standardized Pearson residual.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    
    S, P = X.shape
    disp = max(disp, 1e-10)
    
    # Pearson residuals for negative binomial
    # r = (y - mu) / sqrt(var)
    # var = mu + disp * mu^2
    var_y = mu + disp * mu ** 2
    var_y = np.maximum(var_y, 1e-10)
    
    pearson_resid = (y - mu) / np.sqrt(var_y)
    
    # Weight matrix for NB GLM (W = mu / (1 + disp * mu))
    W = mu / (1 + disp * mu)
    W = np.maximum(W, 1e-10)
    
    # Hat matrix diagonal: h_ii = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}
    # Simplified computation
    W_sqrt = np.sqrt(W)
    XtWX = (X.T * W) @ X
    
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # Singular matrix, return NaN
        return np.full(S, np.nan)
    
    # h_ii = x_i^T (X^T W X)^{-1} x_i * w_i
    leverage = np.zeros(S)
    for i in range(S):
        leverage[i] = W[i] * (X[i, :] @ XtWX_inv @ X[i, :])
    
    # Cook's distance
    # D_i = e_i^2 * h_ii / (p * (1 - h_ii)^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooks_d = (pearson_resid ** 2 * leverage) / (P * (1 - leverage) ** 2)
    
    cooks_d = np.where(np.isfinite(cooks_d), cooks_d, 0.0)
    
    return cooks_d


def detect_outliers(counts, size_factors, dispersions, design_matrix,
                    cutoff=None, percentile=99):
    """
    Detect outlier samples for each gene using Cook's distance.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    size_factors : np.ndarray
        Size factors for each sample.
    dispersions : np.ndarray
        Gene-wise dispersion estimates.
    design_matrix : np.ndarray
        Design matrix (samples x parameters).
    cutoff : float, optional
        Cook's distance cutoff. If None, uses F-distribution-based cutoff.
    percentile : float, default 99
        Percentile of F-distribution to use for cutoff (only if cutoff is None).
        
    Returns
    -------
    np.ndarray
        Boolean matrix (genes x samples) indicating outliers.
    np.ndarray
        Cook's distance matrix (genes x samples).
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.random.poisson(100, (1000, 6))
    >>> sf = np.ones(6)
    >>> disp = np.full(1000, 0.1)
    >>> X = np.column_stack([np.ones(6), [0,0,0,1,1,1]])
    >>> outliers, cooks = detect_outliers(counts, sf, disp, X)
    
    Notes
    -----
    DESeq2 uses Cook's distance to identify samples that have unusually
    large influence on the fitted values for a gene. These are flagged
    as outliers and can optionally be replaced or have their p-values
    set to NA.
    
    The default cutoff is the 99th percentile of the F(p, n-p) distribution,
    where p is the number of parameters and n is the number of samples.
    """
    Y = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    disp = np.asarray(dispersions, dtype=float)
    X = np.asarray(design_matrix, dtype=float)
    
    G, S = Y.shape
    P = X.shape[1]
    
    # Calculate cutoff if not provided
    if cutoff is None:
        # F(p, n-p) distribution, 99th percentile
        df1 = P
        df2 = max(S - P, 1)
        cutoff = f_dist.ppf(percentile / 100, df1, df2)
    
    # Initialize output
    cooks_matrix = np.full((G, S), np.nan)
    outlier_matrix = np.full((G, S), False)
    
    # Log offset for GLM
    offset = np.log(sf + 1e-12)
    
    for g in range(G):
        y = Y[g, :]
        alpha = disp[g]
        
        if y.sum() == 0:
            continue
        
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.1
        
        # Fit GLM to get fitted values
        try:
            fam = sm.families.NegativeBinomial(alpha=alpha)
            model = sm.GLM(y, X, family=fam, offset=offset)
            res = model.fit()
            mu = res.mu
        except Exception:
            continue
        
        # Calculate Cook's distance
        cooks_d = calculate_cooks_distance(y, X, mu, alpha, offset)
        cooks_matrix[g, :] = cooks_d
        
        # Flag outliers
        outlier_matrix[g, :] = cooks_d > cutoff
    
    return outlier_matrix, cooks_matrix


def replace_outliers(counts, outlier_mask, method='trimmed_mean'):
    """
    Replace outlier counts with imputed values.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    outlier_mask : np.ndarray
        Boolean matrix (genes x samples) indicating outliers.
    method : str, default 'trimmed_mean'
        Replacement method. Options:
        - 'trimmed_mean': Replace with trimmed mean of non-outlier samples
        - 'median': Replace with median of non-outlier samples
        - 'na': Replace with NaN (counts will be excluded)
        
    Returns
    -------
    np.ndarray
        Count matrix with outliers replaced.
        
    Examples
    --------
    >>> counts_clean = replace_outliers(counts, outliers, method='trimmed_mean')
    
    Notes
    -----
    DESeq2 by default sets p-values to NA for genes with outliers rather
    than replacing the counts. The 'trimmed_mean' replacement is useful
    when you want to keep all genes in the analysis.
    """
    counts = np.asarray(counts, dtype=float).copy()
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    
    G, S = counts.shape
    
    for g in range(G):
        outliers = outlier_mask[g, :]
        if not outliers.any():
            continue
        
        non_outlier = ~outliers
        if not non_outlier.any():
            # All samples are outliers, can't impute
            if method == 'na':
                counts[g, :] = np.nan
            continue
        
        if method == 'trimmed_mean':
            replacement = np.mean(counts[g, non_outlier])
        elif method == 'median':
            replacement = np.median(counts[g, non_outlier])
        elif method == 'na':
            replacement = np.nan
        else:
            raise ValueError(f"Unknown replacement method: {method}")
        
        counts[g, outliers] = replacement
    
    return counts


def flag_outlier_genes(outlier_mask, max_outliers=None):
    """
    Flag genes that have too many outlier samples.
    
    Parameters
    ----------
    outlier_mask : np.ndarray
        Boolean matrix (genes x samples) indicating outliers.
    max_outliers : int, optional
        Maximum allowed outliers per gene. Default is floor(n_samples / 5).
        
    Returns
    -------
    np.ndarray
        Boolean array (genes,) indicating genes with too many outliers.
        
    Notes
    -----
    Genes with many outlier samples may indicate systematic issues
    (e.g., mapping problems, batch effects) rather than biological
    variation. These genes may need to be filtered or handled specially.
    """
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    G, S = outlier_mask.shape
    
    if max_outliers is None:
        max_outliers = S // 5
    
    outlier_counts = outlier_mask.sum(axis=1)
    
    return outlier_counts > max_outliers
