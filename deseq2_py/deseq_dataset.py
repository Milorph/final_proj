"""
DESeqDataSet container class for managing RNA-seq analysis.

This module provides a container class that holds count data, sample
metadata, and analysis results, providing a unified interface for
running the DESeq2 analysis pipeline.

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
import pandas as pd


class DESeqDataSet:
    """
    Container for DESeq2-style differential expression analysis.
    
    Stores count data, sample metadata, and analysis results in a
    single object that provides methods for running the full pipeline
    and extracting results.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    coldata : pd.DataFrame
        Sample metadata with experimental variables as columns.
        Row names should be sample identifiers matching count columns.
    design : str, default "~ condition"
        R-style formula specifying the model.
        
    Attributes
    ----------
    counts_raw : np.ndarray
        Raw count matrix.
    counts_normalized : np.ndarray or None
        Normalized counts (after calling deseq2() or estimate_size_factors()).
    coldata : pd.DataFrame
        Sample metadata.
    design : str
        Design formula.
    gene_names : np.ndarray
        Gene identifiers.
    sample_names : np.ndarray
        Sample identifiers.
    size_factors : np.ndarray or None
        Estimated size factors.
    dispersions : np.ndarray or None
        Final dispersion estimates.
    results_df : pd.DataFrame or None
        Results from differential expression testing.
        
    Examples
    --------
    >>> from deseq2_py import DESeqDataSet
    >>> 
    >>> # Create from DataFrame
    >>> dds = DESeqDataSet(counts_df, coldata_df, design="~ condition")
    >>> 
    >>> # Run full analysis
    >>> dds.deseq2()
    >>> 
    >>> # Get results
    >>> res = dds.results(alpha=0.05)
    >>> print(res.head())
    
    Notes
    -----
    The DESeqDataSet class is modeled after the R Bioconductor class
    of the same name, providing a Pythonic interface to the DESeq2
    analysis workflow.
    """
    
    def __init__(self, counts, coldata, design="~ condition"):
        """
        Initialize DESeqDataSet.
        
        Parameters
        ----------
        counts : np.ndarray or pd.DataFrame
            Raw count matrix (genes x samples).
        coldata : pd.DataFrame
            Sample metadata.
        design : str
            Design formula.
        """
        # Handle counts
        if isinstance(counts, pd.DataFrame):
            self.gene_names = np.array(counts.index)
            self.sample_names = np.array(counts.columns)
            self.counts_raw = counts.values.astype(float)
        else:
            self.counts_raw = np.asarray(counts, dtype=float)
            self.gene_names = np.array([f"gene_{i}" for i in range(counts.shape[0])])
            self.sample_names = np.array([f"sample_{i}" for i in range(counts.shape[1])])
        
        # Handle coldata
        if not isinstance(coldata, pd.DataFrame):
            raise TypeError("coldata must be a pandas DataFrame")
        self.coldata = coldata
        
        # Parse design formula
        self.design = design
        
        # Validate dimensions
        G, S = self.counts_raw.shape
        if len(coldata) != S:
            raise ValueError(f"Number of samples in coldata ({len(coldata)}) "
                           f"doesn't match counts ({S})")
        
        # Initialize result placeholders
        self.size_factors = None
        self.counts_normalized = None
        self.dispersions = None
        self.dispersions_gw = None
        self.dispersions_trend = None
        self.base_means = None
        self.results_dict = None
        self.results_df = None
        self.design_matrix = None
        self.design_columns = None
        
        # Build design matrix
        self._build_design_matrix()
    
    def _build_design_matrix(self):
        """Build design matrix from formula and coldata."""
        from .design import create_design_matrix
        self.design_matrix, self.design_columns = create_design_matrix(
            self.coldata, self.design)
    
    def estimate_size_factors(self, method='ratio', control_genes=None):
        """
        Estimate size factors for normalization.
        
        Parameters
        ----------
        method : str, default 'ratio'
            Method for estimation ('ratio' or 'poscounts').
        control_genes : np.ndarray, optional
            Indices of control genes to use.
            
        Returns
        -------
        DESeqDataSet
            Self, for method chaining.
        """
        from .size_factors import estimate_size_factors as est_sf
        
        self.size_factors = est_sf(self.counts_raw, type=method,
                                   control_genes=control_genes)
        self.counts_normalized = self.counts_raw / self.size_factors
        
        return self
    
    def estimate_dispersions(self, fit_type='parametric', min_replicates=7):
        """
        Estimate gene-wise dispersions.
        
        Parameters
        ----------
        fit_type : str, default 'parametric'
            Type of dispersion trend fitting ('parametric', 'local', 'mean').
        min_replicates : int, default 7
            Minimum replicates for dispersion estimation.
            
        Returns
        -------
        DESeqDataSet
            Self, for method chaining.
        """
        from .dispersion_optimized import estimate_dispersions as est_disp
        
        if self.size_factors is None:
            self.estimate_size_factors()
        
        (self.base_means, self.dispersions, self.dispersions_gw,
         self.dispersions_trend, _, self._is_outlier) = est_disp(
            self.counts_raw, self.size_factors, self.design_matrix)
        
        return self
    
    def nbinom_wald_test(self, coef_index=1):
        """
        Run negative binomial Wald test.
        
        Parameters
        ----------
        coef_index : int, default 1
            Index of coefficient to test.
            
        Returns
        -------
        DESeqDataSet
            Self, for method chaining.
        """
        from .nbinom_wald import nb_glm_wald
        
        if self.dispersions is None:
            self.estimate_dispersions()
        
        log2_fc, se_log2, wald, pvals, padj = nb_glm_wald(
            self.counts_raw, self.size_factors, self.dispersions,
            self.design_matrix, coef_index=coef_index)
        
        self.results_dict = {
            'baseMean': self.base_means,
            'log2FoldChange': log2_fc,
            'lfcMLE': log2_fc.copy(),
            'lfcSE': se_log2,
            'stat': wald,
            'pvalue': pvals,
            'padj': padj,
            'dispersion': self.dispersions
        }
        
        return self
    
    def deseq2(self, test='Wald', fit_type='parametric', 
               betaPrior=False, quiet=False):
        """
        Run the full DESeq2 analysis pipeline.
        
        This is a convenience method that runs all analysis steps:
        1. Estimate size factors
        2. Estimate dispersions
        3. Run statistical test (Wald or LRT)
        4. Apply LFC shrinkage if betaPrior=True
        
        Parameters
        ----------
        test : str, default 'Wald'
            Statistical test to use ('Wald' or 'LRT').
        fit_type : str, default 'parametric'
            Dispersion trend fitting method.
        betaPrior : bool, default False
            Whether to apply LFC shrinkage.
        quiet : bool, default False
            Whether to suppress progress messages.
            
        Returns
        -------
        DESeqDataSet
            Self, for method chaining.
            
        Examples
        --------
        >>> dds = DESeqDataSet(counts, coldata, design="~ condition")
        >>> dds.deseq2()
        >>> res = dds.results()
        """
        if not quiet:
            print("Estimating size factors...")
        self.estimate_size_factors()
        
        if not quiet:
            print("Estimating dispersions...")
        self.estimate_dispersions(fit_type=fit_type)
        
        if test == 'Wald':
            if not quiet:
                print("Running Wald test...")
            self.nbinom_wald_test()
        elif test == 'LRT':
            if not quiet:
                print("Running LRT...")
            # For LRT, we need reduced design
            # Default: reduced model is intercept only
            self._run_lrt()
        else:
            raise ValueError(f"Unknown test: {test}")
        
        if betaPrior:
            if not quiet:
                print("Applying LFC shrinkage...")
            self._apply_lfc_shrinkage()
        
        if not quiet:
            print("Done.")
        
        return self
    
    def _run_lrt(self, reduced=None):
        """Run likelihood ratio test with optional reduced design."""
        from .lrt import lrt_test
        
        if reduced is None:
            # Default reduced model: intercept only
            reduced_design = np.ones((self.design_matrix.shape[0], 1))
        else:
            from .design import create_design_matrix
            reduced_design, _ = create_design_matrix(self.coldata, reduced)
        
        result = lrt_test(self.counts_raw, self.size_factors, 
                         self.dispersions, self.design_matrix, reduced_design)
        
        # We still need LFC from Wald
        self.nbinom_wald_test()
        
        # Replace p-values with LRT p-values
        self.results_dict['stat'] = result['stat']
        self.results_dict['pvalue'] = result['pvalue']
        self.results_dict['padj'] = result['padj']
    
    def _apply_lfc_shrinkage(self, type='normal'):
        """Apply LFC shrinkage to results."""
        from .lfc_shrinkage import lfcShrink
        
        if self.results_dict is None:
            raise ValueError("Must run statistical test before LFC shrinkage")
        
        lfc_mle = self.results_dict['lfcMLE']
        lfc_se = self.results_dict['lfcSE']
        
        lfc_shrunk = lfcShrink(lfc_mle, lfc_se, type=type)
        self.results_dict['log2FoldChange'] = lfc_shrunk
    
    def results(self, contrast=None, name=None, alpha=0.05,
                lfcThreshold=0.0, altHypothesis='greaterAbs',
                lfcShrink=None):
        """
        Extract results from DESeq2 analysis.
        
        Parameters
        ----------
        contrast : list, optional
            Contrast specification [factor, level1, level2].
        name : str, optional
            Name of coefficient to extract.
        alpha : float, default 0.05
            FDR threshold for independent filtering.
        lfcThreshold : float, default 0.0
            Log2 fold change threshold for testing.
        altHypothesis : str, default 'greaterAbs'
            Alternative hypothesis for LFC threshold testing.
        lfcShrink : str, optional
            Shrinkage method to apply ('apeglm', 'normal', 'ashr').
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with differential expression statistics.
        """
        from .results import results as get_results
        
        if self.results_dict is None:
            raise ValueError("Must run deseq2() before extracting results")
        
        result_df = get_results(
            self.results_dict,
            contrast=contrast,
            name=name,
            alpha=alpha,
            lfc_threshold=lfcThreshold,
            alt_hypothesis=altHypothesis,
            lfcShrink=lfcShrink,
            design_columns=self.design_columns
        )
        
        result_df.index = self.gene_names
        self.results_df = result_df
        
        return result_df
    
    def counts(self, normalized=False):
        """
        Get count matrix.
        
        Parameters
        ----------
        normalized : bool, default False
            Whether to return normalized counts.
            
        Returns
        -------
        pd.DataFrame
            Count matrix with gene names as index and sample names as columns.
        """
        if normalized:
            if self.counts_normalized is None:
                self.estimate_size_factors()
            data = self.counts_normalized
        else:
            data = self.counts_raw
        
        return pd.DataFrame(data, index=self.gene_names, 
                           columns=self.sample_names)
    
    def sizeFactors(self):
        """
        Get size factors.
        
        Returns
        -------
        np.ndarray
            Size factors for each sample.
        """
        if self.size_factors is None:
            self.estimate_size_factors()
        return self.size_factors
    
    def dispersions_gene(self):
        """
        Get gene-wise dispersion estimates.
        
        Returns
        -------
        np.ndarray
            Dispersion estimate for each gene.
        """
        return self.dispersions
    
    def vst(self, blind=True):
        """
        Apply variance stabilizing transformation.
        
        Parameters
        ----------
        blind : bool, default True
            Whether to use blind transformation.
            
        Returns
        -------
        pd.DataFrame
            VST-transformed data.
        """
        from .transformations import vst
        
        if self.size_factors is None:
            self.estimate_size_factors()
        
        transformed = vst(self.counts_raw, self.size_factors, 
                         self.dispersions)
        
        return pd.DataFrame(transformed, index=self.gene_names,
                           columns=self.sample_names)
    
    def rlog(self, blind=True):
        """
        Apply regularized log transformation.
        
        Parameters
        ----------
        blind : bool, default True
            Whether to use blind transformation.
            
        Returns
        -------
        pd.DataFrame
            rlog-transformed data.
        """
        from .transformations import rlog
        
        if self.size_factors is None:
            self.estimate_size_factors()
        
        transformed = rlog(self.counts_raw, self.size_factors,
                          self.dispersions, blind=blind)
        
        return pd.DataFrame(transformed, index=self.gene_names,
                           columns=self.sample_names)
    
    def plotMA(self, alpha=0.05, **kwargs):
        """
        Create MA plot of results.
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance threshold.
        **kwargs
            Additional arguments passed to plotMA().
            
        Returns
        -------
        matplotlib.axes.Axes
            Plot axes.
        """
        from .plotting import plotMA
        
        if self.results_dict is None:
            raise ValueError("Must run deseq2() before plotting")
        
        return plotMA(self.results_dict, alpha=alpha, **kwargs)
    
    def plotDispEsts(self, **kwargs):
        """
        Plot dispersion estimates.
        
        Returns
        -------
        matplotlib.axes.Axes
            Plot axes.
        """
        from .plotting import plotDispEsts
        
        if self.dispersions is None:
            raise ValueError("Must estimate dispersions before plotting")
        
        return plotDispEsts(self.base_means, self.dispersions_gw,
                           self.dispersions_trend, self.dispersions, **kwargs)
    
    def plotPCA(self, ntop=500, color_by=None, **kwargs):
        """
        PCA plot of samples.
        
        Parameters
        ----------
        ntop : int, default 500
            Number of top variable genes to use.
        color_by : str, optional
            Column in coldata to use for coloring.
        **kwargs
            Additional arguments passed to plotPCA().
            
        Returns
        -------
        matplotlib.axes.Axes
            Plot axes.
        """
        from .plotting import plotPCA
        
        vst_data = self.vst()
        return plotPCA(vst_data, self.coldata, color_by=color_by,
                      n_top=ntop, **kwargs)
    
    def summary(self, alpha=0.05):
        """
        Print summary of results.
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance threshold.
        """
        from .results import summary
        
        if self.results_df is None:
            self.results()
        
        return summary(self.results_df, alpha=alpha)
    
    def __repr__(self):
        """String representation."""
        G, S = self.counts_raw.shape
        analyzed = "analyzed" if self.results_dict is not None else "not analyzed"
        return f"DESeqDataSet with {G} genes and {S} samples ({analyzed})"
