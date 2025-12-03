"""
Variance stabilizing transformations for RNA-seq count data.

This module implements three transformations commonly used to prepare
count data for downstream analysis like PCA, clustering, and visualization:
- VST (Variance Stabilizing Transformation)
- rlog (Regularized Log Transformation)
- normTransform (Simple log2 transformation)

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
    - Anders S, Huber W (2010). Differential expression analysis for sequence 
      count data. Genome Biology 11:R106
"""

import numpy as np
import pandas as pd
from scipy.special import polygamma


def vst(counts, size_factors=None, dispersions=None, fit_type='parametric'):
    """
    Variance Stabilizing Transformation.
    
    Transforms count data to approximately homoskedastic values,
    stabilizing variance across the range of mean expression.
    
    The VST uses the relationship between variance and mean in
    negative binomial distributed data:
    
        vst(y) ≈ (2 * arcsinh(sqrt(disp * y)) - log(disp) - log(4)) / log(2)
    
    For genes with low dispersion, this approximates log2(y).
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    size_factors : np.ndarray, optional
        Size factors for each sample. If None, estimated from data.
    dispersions : np.ndarray, optional
        Gene-wise dispersion estimates. If None, estimated from data.
    fit_type : str, default 'parametric'
        Type of dispersion fit. Currently only 'parametric' is supported.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        VST transformed values with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.random.poisson(100, (1000, 6))
    >>> vst_data = vst(counts)
    >>> # Variance is now approximately constant across mean expression
    
    Notes
    -----
    VST is faster than rlog and is recommended for large datasets
    (n > 30 samples). For small datasets, rlog may be preferable.
    
    The transformation is blind to experimental design by default.
    For visualization purposes (PCA), this is often appropriate to
    avoid biasing the analysis.
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape
    
    # Estimate size factors if not provided
    if size_factors is None:
        from .size_factors import estimate_size_factors
        size_factors = estimate_size_factors(counts)
    
    size_factors = np.asarray(size_factors, dtype=float)
    
    # Normalize counts
    norm_counts = counts / size_factors
    
    # Estimate dispersions if not provided
    if dispersions is None:
        # Use a simple moment-based estimator for speed
        mean_counts = norm_counts.mean(axis=1)
        var_counts = norm_counts.var(axis=1, ddof=1)
        
        # dispersion = (var - mean) / mean^2
        with np.errstate(divide='ignore', invalid='ignore'):
            dispersions = np.maximum((var_counts - mean_counts) / (mean_counts ** 2), 1e-8)
        dispersions = np.where(np.isfinite(dispersions), dispersions, 0.1)
    
    dispersions = np.asarray(dispersions, dtype=float)
    
    # Apply VST transformation
    # vst(y) ≈ (2 * arcsinh(sqrt(disp * y)) - log(disp) - log(4)) / log(2)
    # Simplified for numerical stability
    transformed = np.zeros_like(counts)
    
    for i in range(G):
        disp = max(dispersions[i], 1e-8)
        y = norm_counts[i, :]
        
        # Avoid log(0) issues
        y = np.maximum(y, 1e-8)
        
        # Transformation based on NB variance structure
        # For large counts: approaches log2(y)
        # For small counts: regularizes toward 0
        transformed[i, :] = (2 * np.arcsinh(np.sqrt(disp * y)) - 
                            np.log(disp) - np.log(4)) / np.log(2)
    
    if is_df:
        return pd.DataFrame(transformed, index=index, columns=columns)
    return transformed


def rlog(counts, size_factors=None, dispersions=None, blind=True, 
         intercept_only=True, beta_prior_var=1.0):
    """
    Regularized Log Transformation.
    
    Transforms count data using a shrinkage approach that moderates
    log fold changes for genes with low counts, producing values
    that are approximately homoskedastic.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    size_factors : np.ndarray, optional
        Size factors for each sample. If None, estimated from data.
    dispersions : np.ndarray, optional
        Gene-wise dispersion estimates. If None, estimated from data.
    blind : bool, default True
        Whether to use a blind design (intercept only) for transformation.
        Set to False to use experimental design for better separation.
    intercept_only : bool, default True
        If True, use intercept-only model (recommended for visualization).
    beta_prior_var : float, default 1.0
        Prior variance for shrinkage of log fold changes.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        rlog transformed values with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.random.poisson(100, (1000, 6))
    >>> rlog_data = rlog(counts)
    
    Notes
    -----
    rlog is slower than VST but provides better variance stabilization
    for small datasets (n < 30 samples) or datasets with large size
    factor differences between samples.
    
    The rlog transformation uses a shrinkage approach where genes
    with low counts have their log fold changes moderated toward
    the intercept (overall mean).
    
    References:
        Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
        and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape
    
    # Estimate size factors if not provided
    if size_factors is None:
        from .size_factors import estimate_size_factors
        size_factors = estimate_size_factors(counts)
    
    size_factors = np.asarray(size_factors, dtype=float)
    
    # Normalize counts
    norm_counts = counts / size_factors
    
    # Estimate dispersions if not provided
    if dispersions is None:
        mean_counts = norm_counts.mean(axis=1)
        var_counts = norm_counts.var(axis=1, ddof=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dispersions = np.maximum((var_counts - mean_counts) / (mean_counts ** 2), 1e-8)
        dispersions = np.where(np.isfinite(dispersions), dispersions, 0.1)
    
    dispersions = np.asarray(dispersions, dtype=float)
    
    # Apply rlog transformation with shrinkage
    transformed = np.zeros_like(counts)
    
    # Log of size factors for offset
    log_sf = np.log(size_factors)
    
    for i in range(G):
        y = counts[i, :]
        disp = max(dispersions[i], 1e-8)
        
        # Skip genes with all zeros
        if y.sum() == 0:
            transformed[i, :] = 0
            continue
        
        # Mean normalized count
        mu = np.maximum(norm_counts[i, :].mean(), 1e-8)
        
        # Log2 of normalized counts with pseudocount
        log2_norm = np.log2(np.maximum(norm_counts[i, :], 0.5))
        
        # Shrinkage factor based on dispersion and mean
        # Higher dispersion or lower mean = more shrinkage
        # Based on the posterior variance formula
        prior_var = beta_prior_var
        
        # Approximate observed variance of log values
        with np.errstate(divide='ignore', invalid='ignore'):
            obs_var = disp / mu + 1 / mu  # Approximation based on NB variance
        
        # Shrinkage weight (0 = full shrinkage, 1 = no shrinkage)
        shrink_weight = prior_var / (prior_var + obs_var)
        shrink_weight = np.clip(shrink_weight, 0, 1)
        
        # Shrink toward the gene's mean log2 expression
        mean_log2 = np.log2(np.maximum(mu, 0.5))
        transformed[i, :] = shrink_weight * log2_norm + (1 - shrink_weight) * mean_log2
    
    if is_df:
        return pd.DataFrame(transformed, index=index, columns=columns)
    return transformed


def normTransform(counts, size_factors=None, pseudocount=1.0):
    """
    Simple log2 transformation of normalized counts.
    
    This is the simplest variance-stabilizing approach:
    log2(normalized_counts + pseudocount)
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    size_factors : np.ndarray, optional
        Size factors for each sample. If None, estimated from data.
    pseudocount : float, default 1.0
        Value added before log transformation to avoid log(0).
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        log2 transformed normalized values.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([[100, 200], [50, 100]])
    >>> norm_log = normTransform(counts)
    
    Notes
    -----
    This is the fastest transformation but provides less effective
    variance stabilization than VST or rlog, especially for genes
    with low counts.
    
    Suitable for quick exploratory analysis or when computational
    speed is important.
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    
    # Estimate size factors if not provided
    if size_factors is None:
        from .size_factors import estimate_size_factors
        size_factors = estimate_size_factors(counts)
    
    size_factors = np.asarray(size_factors, dtype=float)
    
    # Normalize and transform
    norm_counts = counts / size_factors
    transformed = np.log2(norm_counts + pseudocount)
    
    if is_df:
        return pd.DataFrame(transformed, index=index, columns=columns)
    return transformed
