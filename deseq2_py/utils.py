"""
Utility functions for DESeq2-like analysis.

This module provides helper functions for common RNA-seq normalization
operations including FPKM, TPM, and FPM calculations.

References:
    - Anders S, Huber W (2010). Differential expression analysis for sequence 
      count data. Genome Biology 11:R106
    - Mortazavi A et al. (2008). Mapping and quantifying mammalian transcriptomes 
      by RNA-Seq. Nature Methods 5:621-628
"""

import numpy as np
import pandas as pd


def fpm(counts, size_factors=None):
    """
    Calculate Fragments Per Million (FPM), also known as CPM (Counts Per Million).
    
    This is a simple library-size normalization that scales counts to a 
    common library size of one million reads.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    size_factors : np.ndarray, optional
        Pre-computed size factors. If None, uses library size normalization.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        FPM normalized counts with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([[100, 200], [50, 100], [25, 50]])
    >>> fpm_vals = fpm(counts)
    >>> # Each column sums to 1e6
    
    Notes
    -----
    FPM = (count / library_size) * 1e6
    This does not account for gene length, so is not suitable for 
    comparing expression between genes.
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    
    if size_factors is None:
        # Use library size normalization
        lib_sizes = counts.sum(axis=0)
        normalized = counts / lib_sizes * 1e6
    else:
        size_factors = np.asarray(size_factors, dtype=float)
        # First normalize by size factors, then scale to millions
        normalized = counts / size_factors
        lib_sizes = normalized.sum(axis=0)
        normalized = normalized / lib_sizes * 1e6
    
    if is_df:
        return pd.DataFrame(normalized, index=index, columns=columns)
    return normalized


def fpkm(counts, gene_lengths, size_factors=None):
    """
    Calculate Fragments Per Kilobase per Million (FPKM).
    
    FPKM normalizes for both library size and gene length, allowing
    comparison of expression levels between genes within a sample.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    gene_lengths : np.ndarray
        Gene lengths in base pairs (one per gene).
    size_factors : np.ndarray, optional
        Pre-computed size factors. If None, uses library size normalization.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        FPKM normalized values with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([[100, 200], [50, 100]])
    >>> gene_lengths = np.array([1000, 2000])
    >>> fpkm_vals = fpkm(counts, gene_lengths)
    
    Notes
    -----
    FPKM = (count * 1e9) / (gene_length * library_size)
    
    For paired-end data, FPKM is more appropriate than RPKM.
    Note that FPKM values are not directly comparable across samples
    due to compositional bias.
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    gene_lengths = np.asarray(gene_lengths, dtype=float)
    
    if gene_lengths.shape[0] != counts.shape[0]:
        raise ValueError("gene_lengths must have same length as number of genes")
    
    # Calculate library sizes (use raw counts regardless of size factors for FPKM)
    lib_sizes = counts.sum(axis=0)
    
    # FPKM = (count * 1e9) / (gene_length * library_size)
    # gene_length in bp, so divide by 1000 to get kb
    fpkm_vals = (counts * 1e9) / (gene_lengths[:, np.newaxis] * lib_sizes)
    
    if is_df:
        return pd.DataFrame(fpkm_vals, index=index, columns=columns)
    return fpkm_vals


def tpm(counts, gene_lengths):
    """
    Calculate Transcripts Per Million (TPM).
    
    TPM is preferred over FPKM for cross-sample comparisons because
    the sum of all TPMs is constant across samples (1 million).
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    gene_lengths : np.ndarray
        Gene lengths in base pairs (one per gene).
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        TPM normalized values with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([[100, 200], [50, 100]])
    >>> gene_lengths = np.array([1000, 2000])
    >>> tpm_vals = tpm(counts, gene_lengths)
    >>> # Each column sums to 1e6
    
    Notes
    -----
    TPM calculation:
    1. Divide counts by gene length (in kb) to get reads per kilobase (RPK)
    2. Sum all RPK values in sample to get scaling factor
    3. Divide each RPK by scaling factor and multiply by 1e6
    
    TPM_i = (count_i / length_i) / sum_j(count_j / length_j) * 1e6
    
    References:
        Wagner GP, Kin K, Lynch VJ (2012). Measurement of mRNA abundance 
        using RNA-seq data: RPKM measure is inconsistent among samples. 
        Theory Biosci. 131:281-285
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    gene_lengths = np.asarray(gene_lengths, dtype=float)
    
    if gene_lengths.shape[0] != counts.shape[0]:
        raise ValueError("gene_lengths must have same length as number of genes")
    
    # Step 1: Divide counts by gene length (in kb)
    rpk = counts / (gene_lengths[:, np.newaxis] / 1000)
    
    # Step 2-3: Normalize to sum to 1 million per sample
    scaling_factors = rpk.sum(axis=0)
    tpm_vals = rpk / scaling_factors * 1e6
    
    if is_df:
        return pd.DataFrame(tpm_vals, index=index, columns=columns)
    return tpm_vals


def normalize_counts(counts, size_factors):
    """
    Normalize counts by size factors.
    
    This is the standard DESeq2 normalization method using
    median-of-ratios size factors.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    size_factors : np.ndarray
        Size factors for each sample.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        Normalized counts with same shape as input.
        
    Examples
    --------
    >>> import numpy as np
    >>> from deseq2_py.size_factors import estimate_size_factors
    >>> counts = np.array([[100, 200], [50, 100], [25, 50]])
    >>> sf = estimate_size_factors(counts)
    >>> norm_counts = normalize_counts(counts, sf)
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts = counts.values
    
    counts = np.asarray(counts, dtype=float)
    size_factors = np.asarray(size_factors, dtype=float)
    
    normalized = counts / size_factors
    
    if is_df:
        return pd.DataFrame(normalized, index=index, columns=columns)
    return normalized


def replace_outliers_with_trimmed_mean(counts, threshold=3.0):
    """
    Replace outlier counts with trimmed mean.
    
    For each gene, identifies samples with counts deviating more than
    `threshold` standard deviations from the mean and replaces them
    with the trimmed mean of the remaining samples.
    
    Parameters
    ----------
    counts : np.ndarray
        Raw count matrix (genes x samples).
    threshold : float, default 3.0
        Number of standard deviations for outlier detection.
        
    Returns
    -------
    np.ndarray
        Counts with outliers replaced by trimmed mean.
        
    Notes
    -----
    This is a simple outlier replacement strategy. DESeq2 uses
    Cook's distance for more sophisticated outlier detection.
    """
    counts = np.asarray(counts, dtype=float).copy()
    G, S = counts.shape
    
    for i in range(G):
        row = counts[i, :]
        if row.std() == 0:
            continue
        
        mean_val = row.mean()
        std_val = row.std()
        
        outlier_mask = np.abs(row - mean_val) > threshold * std_val
        if outlier_mask.any():
            # Calculate trimmed mean excluding outliers
            trimmed_mean = row[~outlier_mask].mean()
            counts[i, outlier_mask] = trimmed_mean
    
    return counts


def filter_low_counts(counts, min_count=10, min_samples=None):
    """
    Filter genes with low counts across samples.
    
    Removes genes that don't have at least `min_count` reads in at
    least `min_samples` samples.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    min_count : int, default 10
        Minimum count threshold.
    min_samples : int, optional
        Minimum number of samples that must meet threshold.
        Default is smallest group size or 2.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        Filtered count matrix.
    np.ndarray
        Boolean mask indicating which genes were kept.
        
    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([[100, 200], [1, 2], [50, 100]])
    >>> filtered, mask = filter_low_counts(counts, min_count=10)
    >>> print(mask)  # [True, False, True]
    """
    is_df = isinstance(counts, pd.DataFrame)
    if is_df:
        index = counts.index
        columns = counts.columns
        counts_arr = counts.values
    else:
        counts_arr = np.asarray(counts, dtype=float)
    
    G, S = counts_arr.shape
    
    if min_samples is None:
        min_samples = max(2, S // 2)
    
    # Count how many samples have >= min_count for each gene
    above_threshold = (counts_arr >= min_count).sum(axis=1)
    mask = above_threshold >= min_samples
    
    filtered = counts_arr[mask, :]
    
    if is_df:
        return pd.DataFrame(filtered, index=index[mask], columns=columns), mask
    return filtered, mask
