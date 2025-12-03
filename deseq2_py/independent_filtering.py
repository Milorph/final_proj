"""
Independent filtering to maximize discoveries at given FDR.

This module implements independent filtering, which filters genes by
mean normalized counts to increase the number of significant discoveries
at a given FDR threshold.

References:
    - Bourgon R, Gentleman R, Huber W (2010). Independent filtering 
      increases detection power for high-throughput experiments. 
      PNAS 107(21):9546-9551
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
from scipy.stats import spearmanr


def find_optimal_threshold(base_means, pvalues, alpha=0.05, n_bins=50):
    """
    Find the optimal filtering threshold that maximizes discoveries.
    
    Uses a quantile-based approach to find the threshold on mean
    normalized counts that maximizes the number of significant
    discoveries after FDR correction.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    pvalues : np.ndarray
        Raw p-values per gene.
    alpha : float, default 0.05
        FDR threshold for significance.
    n_bins : int, default 50
        Number of quantile bins to test.
        
    Returns
    -------
    float
        Optimal filtering threshold on mean counts.
    float
        Proportion of genes that pass the optimal threshold.
        
    Examples
    --------
    >>> threshold, prop = find_optimal_threshold(base_means, pvalues, alpha=0.05)
    >>> filtered_idx = base_means >= threshold
    
    Notes
    -----
    The algorithm:
    1. Test different quantile thresholds on mean counts
    2. For each threshold, calculate number of discoveries at FDR < alpha
    3. Select threshold that maximizes discoveries
    
    Independent filtering exploits the fact that genes with very low
    counts have little power to detect differential expression and
    only add to the multiple testing burden.
    """
    base_means = np.asarray(base_means, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    
    # Only consider genes with valid p-values
    valid = np.isfinite(pvalues) & np.isfinite(base_means)
    
    if valid.sum() < 10:
        return 0.0, 1.0
    
    base_means_valid = base_means[valid]
    pvalues_valid = pvalues[valid]
    
    # Test quantile thresholds
    quantiles = np.linspace(0, 0.8, n_bins)
    thresholds = np.quantile(base_means_valid, quantiles)
    
    best_n_sig = 0
    best_threshold = 0.0
    best_prop = 1.0
    
    from .nbinom_wald import benjamini_hochberg
    
    for i, thresh in enumerate(thresholds):
        # Filter genes
        mask = base_means_valid >= thresh
        if mask.sum() < 10:
            continue
        
        # Calculate FDR-adjusted p-values for filtered set
        pvals_filtered = pvalues_valid[mask]
        padj = benjamini_hochberg(pvals_filtered)
        
        # Count discoveries
        n_sig = np.sum(padj < alpha)
        
        if n_sig > best_n_sig:
            best_n_sig = n_sig
            best_threshold = thresh
            best_prop = mask.sum() / len(base_means_valid)
    
    return best_threshold, best_prop


def independent_filtering(base_means, pvalues, alpha=0.05, 
                          filter_fun=None, theta=None):
    """
    Perform independent filtering to increase discovery power.
    
    Filters genes by an independent statistic (mean normalized counts)
    before multiple testing correction, which can increase power to
    detect truly differentially expressed genes.
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene (filter criterion).
    pvalues : np.ndarray
        Raw p-values per gene.
    alpha : float, default 0.05
        FDR threshold for significance.
    filter_fun : callable, optional
        Function to determine filtering threshold. If None, uses
        find_optimal_threshold.
    theta : float, optional
        Fixed filtering threshold. If provided, filter_fun is ignored.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'padj': FDR-adjusted p-values (NaN for filtered genes)
        - 'filter': Boolean mask of genes passing filter
        - 'threshold': Filtering threshold used
        - 'n_filtered': Number of genes removed
        - 'n_significant': Number of significant discoveries
        
    Examples
    --------
    >>> result = independent_filtering(base_means, pvalues, alpha=0.05)
    >>> significant = result['padj'] < 0.05
    >>> print(f"Found {result['n_significant']} significant genes")
    
    Notes
    -----
    The filter criterion must be independent of the test statistic
    under the null hypothesis. Mean expression level satisfies this
    requirement because it's determined before differential expression
    testing.
    
    References:
        Bourgon R, Gentleman R, Huber W (2010). Independent filtering 
        increases detection power for high-throughput experiments. 
        PNAS 107(21):9546-9551
    """
    from .nbinom_wald import benjamini_hochberg
    
    base_means = np.asarray(base_means, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    
    G = len(pvalues)
    
    # Determine threshold
    if theta is not None:
        threshold = theta
    elif filter_fun is not None:
        threshold = filter_fun(base_means, pvalues)
    else:
        threshold, _ = find_optimal_threshold(base_means, pvalues, alpha)
    
    # Apply filter
    filter_mask = base_means >= threshold
    
    # Initialize adjusted p-values with NaN
    padj = np.full(G, np.nan)
    
    # Calculate adjusted p-values for filtered genes
    valid = filter_mask & np.isfinite(pvalues)
    if valid.sum() > 0:
        padj_filtered = benjamini_hochberg(pvalues[valid])
        padj[valid] = padj_filtered
    
    n_significant = np.sum(padj < alpha)
    
    return {
        'padj': padj,
        'filter': filter_mask,
        'threshold': threshold,
        'n_filtered': G - filter_mask.sum(),
        'n_significant': n_significant
    }


def filter_by_expression(counts, size_factors=None, min_count=10, 
                         min_samples=None, min_mean=None):
    """
    Filter genes by expression level.
    
    A simpler alternative to independent filtering that uses fixed
    thresholds rather than optimizing for discoveries.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    size_factors : np.ndarray, optional
        Size factors for each sample. If None, uses library size.
    min_count : int, default 10
        Minimum count in at least min_samples samples.
    min_samples : int, optional
        Minimum number of samples that must meet min_count threshold.
        Default is smallest group size.
    min_mean : float, optional
        Minimum mean normalized count. If provided, overrides min_count.
        
    Returns
    -------
    np.ndarray
        Boolean mask indicating genes that pass filter.
        
    Examples
    --------
    >>> keep = filter_by_expression(counts, min_count=10)
    >>> counts_filtered = counts[keep, :]
    """
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape
    
    if size_factors is None:
        size_factors = counts.sum(axis=0) / np.median(counts.sum(axis=0))
    
    size_factors = np.asarray(size_factors, dtype=float)
    norm_counts = counts / size_factors
    
    if min_mean is not None:
        # Filter by mean expression
        base_means = norm_counts.mean(axis=1)
        return base_means >= min_mean
    
    if min_samples is None:
        min_samples = max(2, S // 4)
    
    # Filter by count threshold
    above_threshold = (counts >= min_count).sum(axis=1)
    return above_threshold >= min_samples
