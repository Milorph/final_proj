"""
Results and contrast specification for DESeq2-like analysis.

This module provides functions for extracting and formatting results
from differential expression analysis, including contrast specification
and alternative hypothesis testing.

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def results(deseq_result, contrast=None, name=None, alpha=0.05, 
            lfc_threshold=0.0, alt_hypothesis='greaterAbs',
            lfcShrink=None, independent_filtering=True,
            design_columns=None):
    """
    Extract and format results from DESeq2 analysis.
    
    Parameters
    ----------
    deseq_result : dict
        Output from run_deseq containing baseMean, log2FoldChange, etc.
    contrast : list or np.ndarray, optional
        Contrast specification. Can be:
        - List of three strings: [factor, level1, level2] for level1 vs level2
        - Numeric contrast vector
        If None, uses default coefficient from original analysis.
    name : str, optional
        Name of coefficient to test (alternative to contrast).
    alpha : float, default 0.05
        FDR threshold for significance.
    lfc_threshold : float, default 0.0
        Log2 fold change threshold for significance testing.
        If > 0, tests whether |LFC| > threshold rather than |LFC| > 0.
    alt_hypothesis : str, default 'greaterAbs'
        Alternative hypothesis. One of:
        - 'greaterAbs': |LFC| > threshold (default)
        - 'lessAbs': |LFC| < threshold
        - 'greater': LFC > threshold
        - 'less': LFC < -threshold
    lfcShrink : str, optional
        Shrinkage method to apply ('apeglm', 'normal', 'ashr', or None).
    independent_filtering : bool, default True
        Whether to apply independent filtering.
    design_columns : list, optional
        Column names of design matrix (for contrast parsing).
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame with columns:
        - baseMean: Mean normalized count
        - log2FoldChange: Log2 fold change (shrunken if applicable)
        - lfcSE: Standard error of LFC
        - stat: Test statistic
        - pvalue: Raw p-value
        - padj: BH-adjusted p-value
        
    Examples
    --------
    >>> # Basic usage
    >>> res = results(deseq_output, alpha=0.05)
    >>> 
    >>> # With contrast specification
    >>> res = results(deseq_output, contrast=['condition', 'treated', 'control'])
    >>> 
    >>> # With LFC threshold testing
    >>> res = results(deseq_output, lfc_threshold=1.0, alt_hypothesis='greaterAbs')
    
    Notes
    -----
    LFC threshold testing changes the null hypothesis from:
        H0: LFC = 0 vs H1: LFC != 0
    To:
        H0: |LFC| <= threshold vs H1: |LFC| > threshold
    
    This is useful for finding genes with biologically meaningful
    effect sizes, not just statistically significant differences.
    """
    from .nbinom_wald import benjamini_hochberg
    
    # Extract core statistics
    base_mean = np.asarray(deseq_result.get('baseMean', []))
    log2_fc = np.asarray(deseq_result.get('log2FoldChange', 
                         deseq_result.get('lfcMLE', [])))
    lfc_se = np.asarray(deseq_result.get('lfcSE', []))
    stat = np.asarray(deseq_result.get('stat', []))
    pvalue = np.asarray(deseq_result.get('pvalue', []))
    
    G = len(base_mean)
    
    # Apply LFC shrinkage if requested
    if lfcShrink is not None:
        from .lfc_shrinkage import lfcShrink as shrink_func
        lfc_mle = np.asarray(deseq_result.get('lfcMLE', log2_fc))
        log2_fc = shrink_func(lfc_mle, lfc_se, type=lfcShrink)
    
    # Calculate p-values for LFC threshold testing
    if lfc_threshold > 0:
        pvalue = lfc_threshold_pvalue(log2_fc, lfc_se, lfc_threshold, 
                                       alt_hypothesis)
        stat = log2_fc / lfc_se
    
    # Apply independent filtering
    if independent_filtering:
        from .independent_filtering import independent_filtering as ind_filt
        filt_result = ind_filt(base_mean, pvalue, alpha=alpha)
        padj = filt_result['padj']
    else:
        pvalue_clean = np.where(np.isfinite(pvalue), pvalue, 1.0)
        padj = benjamini_hochberg(pvalue_clean)
    
    # Build results DataFrame
    result_df = pd.DataFrame({
        'baseMean': base_mean,
        'log2FoldChange': log2_fc,
        'lfcSE': lfc_se,
        'stat': stat,
        'pvalue': pvalue,
        'padj': padj
    })
    
    return result_df


def lfc_threshold_pvalue(log2_fc, lfc_se, threshold, alt_hypothesis='greaterAbs'):
    """
    Calculate p-values for LFC threshold testing.
    
    Tests whether the true log2 fold change exceeds a specified threshold
    rather than testing whether it's different from zero.
    
    Parameters
    ----------
    log2_fc : np.ndarray
        Log2 fold change estimates.
    lfc_se : np.ndarray
        Standard errors of log2 fold changes.
    threshold : float
        Log2 fold change threshold (positive value).
    alt_hypothesis : str
        Alternative hypothesis type.
        
    Returns
    -------
    np.ndarray
        P-values for threshold test.
    """
    log2_fc = np.asarray(log2_fc, dtype=float)
    lfc_se = np.asarray(lfc_se, dtype=float)
    
    G = len(log2_fc)
    pvals = np.ones(G)
    
    for i in range(G):
        lfc = log2_fc[i]
        se = lfc_se[i]
        
        if not np.isfinite(lfc) or not np.isfinite(se) or se <= 0:
            continue
        
        if alt_hypothesis == 'greaterAbs':
            # H0: |LFC| <= threshold vs H1: |LFC| > threshold
            # P-value is probability that |LFC| <= threshold
            # Use two-sided test
            if lfc >= 0:
                z = (lfc - threshold) / se
                p = norm.sf(z)  # P(Z > z)
            else:
                z = (-lfc - threshold) / se
                p = norm.sf(z)
            pvals[i] = min(2 * p, 1.0)  # Two-sided
            
        elif alt_hypothesis == 'lessAbs':
            # H0: |LFC| >= threshold vs H1: |LFC| < threshold
            z_upper = (threshold - lfc) / se
            z_lower = (-threshold - lfc) / se
            p = norm.cdf(z_upper) - norm.cdf(z_lower)
            pvals[i] = 1 - p
            
        elif alt_hypothesis == 'greater':
            # H0: LFC <= threshold vs H1: LFC > threshold
            z = (lfc - threshold) / se
            pvals[i] = norm.sf(z)
            
        elif alt_hypothesis == 'less':
            # H0: LFC >= -threshold vs H1: LFC < -threshold
            z = (lfc + threshold) / se
            pvals[i] = norm.cdf(z)
            
        else:
            raise ValueError(f"Unknown alt_hypothesis: {alt_hypothesis}")
    
    return pvals


def make_contrast_vector(design_columns, contrast):
    """
    Create numeric contrast vector from contrast specification.
    
    Parameters
    ----------
    design_columns : list
        Column names of design matrix.
    contrast : list or tuple
        Contrast specification: [factor, numerator, denominator].
        
    Returns
    -------
    np.ndarray
        Numeric contrast vector.
        
    Examples
    --------
    >>> cols = ['Intercept', 'condition[T.treated]']
    >>> vec = make_contrast_vector(cols, ['condition', 'treated', 'control'])
    >>> print(vec)  # [0, 1]
    """
    if isinstance(contrast, np.ndarray):
        return contrast
    
    if isinstance(contrast, (list, tuple)) and len(contrast) == 3:
        factor, numerator, denominator = contrast
        
        vec = np.zeros(len(design_columns), dtype=float)
        
        # Find column matching numerator level
        patterns = [
            f"{factor}[T.{numerator}]",
            f"{factor}[{numerator}]",
            f"C({factor})[T.{numerator}]",
            numerator  # Simple name match
        ]
        
        for i, col in enumerate(design_columns):
            for pattern in patterns:
                if pattern in col:
                    vec[i] = 1.0
                    return vec
        
        raise ValueError(f"Could not find column for contrast {contrast}")
    
    raise ValueError(f"Invalid contrast format: {contrast}")


def summary(result_df, alpha=0.05, lfc_cutoff=0.0):
    """
    Print summary of differential expression results.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Results DataFrame from results() function.
    alpha : float, default 0.05
        FDR threshold for significance.
    lfc_cutoff : float, default 0.0
        Optional LFC cutoff for reporting.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    padj = result_df['padj'].values
    lfc = result_df['log2FoldChange'].values
    
    valid = np.isfinite(padj)
    significant = (padj < alpha) & valid
    
    up = significant & (lfc > lfc_cutoff)
    down = significant & (lfc < -lfc_cutoff)
    
    summary_dict = {
        'total_genes': len(result_df),
        'genes_tested': valid.sum(),
        'significant': significant.sum(),
        'upregulated': up.sum(),
        'downregulated': down.sum(),
        'alpha': alpha,
        'lfc_cutoff': lfc_cutoff
    }
    
    print(f"\nDESeq2 Results Summary")
    print(f"=" * 40)
    print(f"Total genes:        {summary_dict['total_genes']}")
    print(f"Genes tested:       {summary_dict['genes_tested']}")
    print(f"Significant (padj < {alpha}): {summary_dict['significant']}")
    print(f"  - Upregulated:    {summary_dict['upregulated']}")
    print(f"  - Downregulated:  {summary_dict['downregulated']}")
    
    return summary_dict
