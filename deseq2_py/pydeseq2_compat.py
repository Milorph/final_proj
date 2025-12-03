"""
PyDESeq2 integration and compatibility layer.

This module provides optional integration with PyDESeq2 (owkin/PyDESeq2,
MIT licensed), falling back to native implementation when PyDESeq2
is not installed.

References:
    - PyDESeq2: https://github.com/owkin/PyDESeq2
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import warnings

# Try to import PyDESeq2
_PYDESEQ2_AVAILABLE = False
try:
    from pydeseq2.dds import DeseqDataSet as PyDeseqDataSet
    from pydeseq2.ds import DeseqStats as PyDeseqStats
    _PYDESEQ2_AVAILABLE = True
except ImportError:
    _PYDESEQ2_AVAILABLE = False


def is_pydeseq2_available():
    """
    Check if PyDESeq2 is installed.
    
    Returns
    -------
    bool
        True if PyDESeq2 is available.
        
    Examples
    --------
    >>> from deseq2_py.pydeseq2_compat import is_pydeseq2_available
    >>> if is_pydeseq2_available():
    ...     print("Using PyDESeq2 backend")
    """
    return _PYDESEQ2_AVAILABLE


class PyDESeq2Wrapper:
    """
    Wrapper around PyDESeq2 to provide consistent API.
    
    This class provides a unified interface that can use either
    PyDESeq2 (when available) or the native implementation.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix (genes x samples) as DataFrame.
    coldata : pd.DataFrame
        Sample metadata.
    design_factors : str or list
        Factor(s) to include in the design formula.
    use_pydeseq2 : bool, default True
        Whether to use PyDESeq2 if available.
        
    Attributes
    ----------
    backend : str
        Which backend is being used ('pydeseq2' or 'native').
    dds : object
        The underlying dataset object.
        
    Examples
    --------
    >>> from deseq2_py.pydeseq2_compat import PyDESeq2Wrapper
    >>> import pandas as pd
    >>> 
    >>> # Create wrapper
    >>> wrapper = PyDESeq2Wrapper(counts_df, coldata_df, design_factors='condition')
    >>> 
    >>> # Run analysis
    >>> wrapper.deseq2()
    >>> 
    >>> # Get results
    >>> res = wrapper.results()
    
    Notes
    -----
    The wrapper attempts to provide a consistent API regardless of
    which backend is used. Some advanced features may only be available
    with PyDESeq2.
    """
    
    def __init__(self, counts, coldata, design_factors='condition',
                 use_pydeseq2=True):
        """Initialize wrapper with data."""
        import pandas as pd
        import numpy as np
        
        self.counts = counts
        self.coldata = coldata
        
        # Determine which backend to use
        if use_pydeseq2 and _PYDESEQ2_AVAILABLE:
            self.backend = 'pydeseq2'
            self._init_pydeseq2(design_factors)
        else:
            self.backend = 'native'
            self._init_native(design_factors)
    
    def _init_pydeseq2(self, design_factors):
        """Initialize PyDESeq2 backend."""
        # PyDESeq2 expects counts as samples x genes
        counts_T = self.counts.T
        
        self.dds = PyDeseqDataSet(
            counts=counts_T,
            metadata=self.coldata,
            design_factors=design_factors
        )
        self.stats = None
    
    def _init_native(self, design_factors):
        """Initialize native backend."""
        from .deseq_dataset import DESeqDataSet
        
        if isinstance(design_factors, list):
            formula = "~ " + " + ".join(design_factors)
        else:
            formula = f"~ {design_factors}"
        
        self.dds = DESeqDataSet(self.counts, self.coldata, design=formula)
    
    def deseq2(self, **kwargs):
        """
        Run the DESeq2 pipeline.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to the backend.
            
        Returns
        -------
        PyDESeq2Wrapper
            Self, for method chaining.
        """
        if self.backend == 'pydeseq2':
            self.dds.deseq2()
        else:
            self.dds.deseq2(**kwargs)
        
        return self
    
    def results(self, contrast=None, alpha=0.05, **kwargs):
        """
        Get differential expression results.
        
        Parameters
        ----------
        contrast : list, optional
            Contrast specification for PyDESeq2.
        alpha : float, default 0.05
            Significance threshold.
        **kwargs
            Additional arguments passed to the backend.
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame.
        """
        import pandas as pd
        
        if self.backend == 'pydeseq2':
            if self.stats is None:
                self.stats = PyDeseqStats(self.dds, contrast=contrast, alpha=alpha)
                self.stats.summary()
            return self.stats.results_df
        else:
            return self.dds.results(contrast=contrast, alpha=alpha, **kwargs)
    
    def vst(self):
        """
        Apply variance stabilizing transformation.
        
        Returns
        -------
        pd.DataFrame
            VST-transformed data.
        """
        if self.backend == 'pydeseq2':
            from pydeseq2.preprocessing import deseq2_norm_transform
            return deseq2_norm_transform(self.dds)
        else:
            return self.dds.vst()
    
    def summary(self):
        """Print analysis summary."""
        if self.backend == 'pydeseq2':
            if self.stats is not None:
                return self.stats.summary()
        else:
            return self.dds.summary()
    
    def __repr__(self):
        """String representation."""
        return f"PyDESeq2Wrapper (backend: {self.backend})"


def run_deseq2_analysis(counts, coldata, design='condition', 
                        use_pydeseq2=True, alpha=0.05, **kwargs):
    """
    Convenience function to run full DESeq2 analysis.
    
    Automatically selects the best available backend and runs
    the complete analysis pipeline.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix (genes x samples).
    coldata : pd.DataFrame
        Sample metadata.
    design : str, default 'condition'
        Design factor(s).
    use_pydeseq2 : bool, default True
        Whether to prefer PyDESeq2 if available.
    alpha : float, default 0.05
        Significance threshold.
    **kwargs
        Additional arguments passed to deseq2().
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame with differential expression statistics.
        
    Examples
    --------
    >>> from deseq2_py.pydeseq2_compat import run_deseq2_analysis
    >>> 
    >>> results = run_deseq2_analysis(counts_df, coldata_df, 
    ...                               design='condition', alpha=0.05)
    >>> significant = results[results['padj'] < 0.05]
    """
    wrapper = PyDESeq2Wrapper(counts, coldata, design_factors=design,
                              use_pydeseq2=use_pydeseq2)
    wrapper.deseq2(**kwargs)
    return wrapper.results(alpha=alpha)


def compare_backends(counts, coldata, design='condition', alpha=0.05):
    """
    Compare results between PyDESeq2 and native implementation.
    
    Useful for validating the native implementation against PyDESeq2.
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix (genes x samples).
    coldata : pd.DataFrame
        Sample metadata.
    design : str, default 'condition'
        Design factor.
    alpha : float, default 0.05
        Significance threshold.
        
    Returns
    -------
    dict
        Comparison metrics including correlations and overlaps.
        
    Examples
    --------
    >>> from deseq2_py.pydeseq2_compat import compare_backends
    >>> 
    >>> if is_pydeseq2_available():
    ...     comparison = compare_backends(counts_df, coldata_df)
    ...     print(f"LFC correlation: {comparison['lfc_correlation']:.4f}")
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    
    if not _PYDESEQ2_AVAILABLE:
        warnings.warn("PyDESeq2 not available, cannot compare backends")
        return None
    
    # Run PyDESeq2
    wrapper_py = PyDESeq2Wrapper(counts, coldata, design_factors=design,
                                  use_pydeseq2=True)
    wrapper_py.deseq2()
    res_pydeseq2 = wrapper_py.results(alpha=alpha)
    
    # Run native
    wrapper_native = PyDESeq2Wrapper(counts, coldata, design_factors=design,
                                      use_pydeseq2=False)
    wrapper_native.deseq2()
    res_native = wrapper_native.results(alpha=alpha)
    
    # Compare results
    # Align genes
    common_genes = set(res_pydeseq2.index) & set(res_native.index)
    
    res_py = res_pydeseq2.loc[list(common_genes)]
    res_na = res_native.loc[list(common_genes)]
    
    # LFC correlation
    lfc_py = res_py['log2FoldChange'].values
    lfc_na = res_na['log2FoldChange'].values
    valid = np.isfinite(lfc_py) & np.isfinite(lfc_na)
    lfc_corr, _ = pearsonr(lfc_py[valid], lfc_na[valid])
    
    # Significant gene overlap
    sig_py = set(res_py[res_py['padj'] < alpha].index)
    sig_na = set(res_na[res_na['padj'] < alpha].index)
    
    if len(sig_py) > 0 and len(sig_na) > 0:
        overlap = len(sig_py & sig_na) / len(sig_py | sig_na)
    else:
        overlap = 0.0
    
    return {
        'n_genes': len(common_genes),
        'lfc_correlation': lfc_corr,
        'n_significant_pydeseq2': len(sig_py),
        'n_significant_native': len(sig_na),
        'significant_overlap': overlap,
        'pydeseq2_results': res_pydeseq2,
        'native_results': res_native
    }
